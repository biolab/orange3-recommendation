import re
import functools
from copy import copy
from itertools import count

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, StringVariable, Variable
from skfusion import fusion


GENERATE_OTYPE = (fusion.ObjectType('LatentSpace' + str(i)) for i in count())


class Relation(Table):
    """Wrapper for `skfusion.fusion.Relation`
    """

    def __new__(cls, *args, **kwargs):
        """Bypass Table.__new__."""
        return object.__new__(Relation)

    def __init__(self, relation):
        """Create a wrapper for fusion.Relation.

        Parameters:
        -----------
        relation: An instance of `skfusion.fusion.Relation`
        """
        self.relation = relation
        meta_vars, self.metas = self._create_metas(relation)
        self._Y = self.W = np.zeros((len(relation.data), 0))

        if relation.col_names is not None:
            attr_names = relation.col_names
        else:
            attr_names = range(relation.data.shape[1])
        self.domain = Domain([ContinuousVariable(name)
                              for name in map(str, attr_names)],
                             metas=meta_vars)
        Table._init_ids(self)

    @staticmethod
    def _create_metas(relation):
        metas = []
        metas_data = [[] for x in relation.data]
        if relation.row_metadata is not None:
            metadata_names = set()
            for md in relation.row_metadata:
                metadata_names.update(md.keys())
            metadata_names = sorted(metadata_names, key=str)

            metas.extend(metadata_names)
            for md, v in zip(metas_data, relation.row_metadata):
                for k in metadata_names:
                    md.append(v.get(k, np.nan))
        elif relation.row_names is not None:
            metas = [relation.row_type.name]
            metas_data = [[name] for name in relation.row_names]

        def create_var(x):
            if isinstance(x, Variable):
                return x
            else:
                return StringVariable(str(x))

        metas_vars = [create_var(x) for x in metas]
        metas = np.array(metas_data, dtype='object')
        return metas_vars, metas

    @property
    def col_type(self):
        """ Column object type"""
        return self.relation.col_type

    @property
    def row_type(self):
        """Row object type"""
        return self.relation.row_type

    @property
    def name(self):
        """Relation name"""
        return self.relation.name

    @property
    @functools.lru_cache()
    def X(self):
        """
        :return: relation data
        """
        data = self.relation.data
        if np.ma.is_masked(data):
            mask = data.mask
            data = data.data.copy()
            data[mask] = np.nan
        else:
            data = data.copy()
        return data

    def __len__(self):
        """Number of rows in relation"""
        return len(self.relation.data)

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        return Table.from_table(domain, source, row_indices)

    @classmethod
    def create(cls, data, row_type, col_type, graph=None):
        row_names = row_metadata = col_names = col_metadata = None
        if row_type:
            row_names = graph.get_names(row_type)
            row_metadata = graph.get_metadata(row_type)
        else:
            row_type = next(GENERATE_OTYPE)
        if col_type:
            col_names = graph.get_names(col_type)
            col_metadata = graph.get_metadata(row_type)
        else:
            col_type = next(GENERATE_OTYPE), None
        return Relation(fusion.Relation(data, row_type, col_type, row_names=row_names, row_metadata=row_metadata,
                                        col_names=col_names, col_metadata=col_metadata))


class RelationCompleter:
    @property
    def name(self):
        """"Completer name"""
        raise NotImplementedError()

    def retrain(self):
        """Retrain the completer using different random seed.

        Returns a new instance of the Completer.
        """
        raise NotImplementedError()

    def can_complete(self, relation):
        """Return True, if the completer had sufficient data to complete the
        given relation.
        """
        raise NotImplementedError()

    def complete(self, relation):
        """Return a completed relation.

         All masked values from the parameter relation should be replaced with the predicted ones.
         """
        raise NotImplementedError()


class FusionGraph:
    def __init__(self, fusion_graph):
        """Wrapper for skfusion FusionGraph

        :type fusion_graph: skfusion.fusion.FusionGraph
        """
        super().__init__()
        self._fusion_graph = fusion_graph

    def __getattr__(self, attr):
        return getattr(self._fusion_graph, attr)

    def get_selected_nodes(self, element_id):
        """ Return ObjectTypes from that correspond to selected `element_id`
            (in the SVG).
        """
        assert element_id.startswith('edge ') or element_id.startswith('node ')
        selected_is_edge = element_id.startswith('edge ')
        # Assumes SVG element's id attributes specify nodes `-delimited
        node_names = re.findall('`([^`]+)`', element_id)
        nodes = [self._fusion_graph.get_object_type(name) for name in node_names]
        assert len(nodes) == 2 if selected_is_edge else len(nodes) == 1
        return nodes


class FittedFusionGraph(FusionGraph, RelationCompleter):
    def __init__(self, fusion_fit):
        """Wrapper for skfusion Fusion Fit

        :type fusion_fit: skfusion.fusion.FusionFit
        """
        super().__init__(fusion_fit.fusion_graph)
        self._fusion_fit = fusion_fit

    @property
    def name(self):
        return (getattr(self._fusion_fit, 'name', '') or
                '{cls}(factors={fac}, iter={mi}, init={it})'.format(
                    cls=self._fusion_fit.__class__.__name__,
                    fac=len(self._fusion_fit.factors_),
                    mi=self._fusion_fit.max_iter,
                    it=self._fusion_fit.init_type))

    @property
    def backbones_(self):
        return self._fusion_fit.backbones_

    @property
    def factors_(self):
        return self._fusion_fit.factors_

    def backbone(self, relation):
        return self._fusion_fit.backbone(relation)

    def factor(self, object_type):
        return self._fusion_fit.factor(object_type)

    def compute_chain(self, chain, end_in_input_space=True):
        row_type = chain[0].row_type
        result = self._fusion_fit.factor(row_type)
        for rel in chain:
            result = np.dot(result, self._fusion_fit.backbone(rel))
        col_type = None
        if end_in_input_space:
            col_type = chain[-1].col_type
            result = np.dot(result, self._fusion_fit.factor(col_type).T)
        return Relation.create(result, row_type, col_type, self)

    # Relation Completer members
    def can_complete(self, relation):
        try:
            for fuser_relation in self.get_relations(relation.row_type,
                                                     relation.col_type):
                if fuser_relation._id == relation._id:
                    return True
        except fusion.DataFusionError:
            pass  # relation.{row,col}_type not in fusion graph
        return False

    def complete(self, relation):
        return self._fusion_fit.complete(relation)

    def retrain(self):
        fg = copy(self._fusion_fit)
        fg.fuse(self._fusion_graph)
        return FittedFusionGraph(fg)
