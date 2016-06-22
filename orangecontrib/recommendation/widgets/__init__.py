import sysconfig
# Category metadata.

# Category icon show in the menu
ICON = "icons/star2.svg"

# Background color for category background in menu
# and widget icon background in workflow.
BACKGROUND = "light-blue"

# Location of widget help files.
WIDGET_HELP_PATH = (
    # Development documentation
    # You need to build help pages manually using
    # make htmlhelp
    # inside doc folder
    ("{DEVELOP_ROOT}/doc/build/htmlhelp/index.html", None),

    # Documentation included in wheel
    # Correct DATA_FILES entry is needed in setup.py and documentation has to be built
    # before the wheel is created.
    ("{}/help/orange3-example/index.html".format(sysconfig.get_path("data")), None),

    # Online documentation url, used when the local documentation is not available.
    # Url should point to a page with a section Widgets. This section should
    # includes links to documentation pages of each widget. Matching is
    # performed by comparing link caption to widget name.
    ("http://orange3-recommendation.readthedocs.io/en/latest", "")
)


"""
***************************************************************************
************************** CREDITS FOR THE ICONS **************************
***************************************************************************

- 'star.svg' icon made by [Freepik] from [www.flaticon.com]
- 'starred-list' icon made by [Freepik] from [www.flaticon.com]
- 'customer.svg' icon made by [Freepik] from [www.flaticon.com]
- 'stars.svg' icon made by [ Alfredo Hernandez] from [www.flaticon.com]
- 'star2.svg' icon made by [EpicCoders] from [www.flaticon.com]
- 'brismf.svg' icon made by [Freepik] from [www.flaticon.com]
- 'manager.svg' icon made by [Freepik] from [www.flaticon.com]
- 'ranking.svg' icon made by [Freepik] from [www.flaticon.com]
- 'candidates-ranking-graphic.svg' icon made by [Freepik] from [www.flaticon.com]

- 'organization.svg' icon made by [Freepik] from [www.flaticon.com]
- 'task.svg' icon made by [Freepik] from [www.flaticon.com]
- 'list.svg' icon made by [Freepik] from [www.flaticon.com]
- 'ranking.svg' icon made by [Freepik] from [www.flaticon.com]
- 'ranking.svg' icon made by [Freepik] from [www.flaticon.com]
"""