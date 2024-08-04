"""
Adapted from typescript via chat-gpt - https://github.com/vega/vega-themes/blob/main/src/theme-vox.ts

Examples:
```
from Scripts.alt_themes import vox_theme

alt.themes.register('vox_theme', lambda: vox_theme)
alt.themes.enable('vox_theme')
```
"""

mark_color = '#3e5c69'
vox_theme = {
    'config': {
        'background': '#fff',

        'arc': {'fill': mark_color},
        'area': {'fill': mark_color},
        'line': {'stroke': mark_color},
        'path': {'stroke': mark_color},
        'rect': {'fill': mark_color},
        'shape': {'stroke': mark_color},
        'symbol': {'fill': mark_color},

        'axis': {
            'domainWidth': 0.5,
            'grid': True,
            'labelPadding': 2,
            'tickSize': 5,
            'tickWidth': 0.5,
            'titleFontWeight': 'normal',
        },

        'axisBand': {
            'grid': False,
        },

        'axisX': {
            'gridWidth': 0.2,
        },

        'axisY': {
            'gridDash': [3],
            'gridWidth': 0.4,
        },

        'legend': {
            'labelFontSize': 11,
            'padding': 1,
            'symbolType': 'square',
        },

        'range': {
            'category': ['#3e5c69', '#6793a6', '#182429', '#0570b0', '#3690c0', '#74a9cf', '#a6bddb', '#e2ddf2'],
        },
    }
}
