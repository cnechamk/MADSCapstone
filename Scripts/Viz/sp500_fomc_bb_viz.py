import pandas as pd
import altair as alt

selection = alt.selection_interval(bind='scales')

def _to_datetime_filter(df, date_col='date', filter_=None):
    df[date_col] = pd.to_datetime(df[date_col])
    if filter_ is not None:
        df = filter_(df, date_col)

    return df


GE_1970 = lambda df, date_col: df.loc[df[date_col] >= pd.to_datetime("1970-01-01")]


p = "Data/sp500.csv"
df_sp500 = pd.read_csv(p)
df_sp500 = _to_datetime_filter(df_sp500, filter_=GE_1970)

sp500_chart = alt.Chart(df_sp500).mark_line().encode(
    x=alt.X('date:T').axis(title="Date"),
    y=alt.Y("close:Q", scale=alt.Scale(type='log', base=10)).axis(title="SP500 Closing Price"),
    color=alt.ColorValue("#c8c8c8")
).properties(
    width=1000,
    height=500
).add_params(
    selection
)

# ========== Ticks ==========
# BEIGE BOOK
# chat gpt helped with the conditional layering of the tooltips

p = "Data/beige_books_kws.csv"
df_bb = pd.read_csv(p)
df_bb.date = pd.to_datetime(df_bb.date)

df_bb = df_bb.pivot_table(
    index='date',
    values='keywords',
    columns='district',
    aggfunc=lambda x: x.str.strip("'[]'")
)

districts = df_bb.columns
df_bb = df_bb.reset_index()

df_bb.date = pd.to_datetime(df_bb.date)
df_bb = _to_datetime_filter(df_bb, filter_=GE_1970)
df_bb = df_bb.merge(df_sp500, how='left', on='date')

# Create the binding and parameter for the checkbox
bind_checkbox = alt.binding_checkbox(name='Add Beige Book Ticks (beige): ')
bb_checkbox = alt.param(bind=bind_checkbox)

# Chart with the tooltip
bb_chart_with_tooltip = alt.Chart(df_bb).mark_tick(orient='vertical', thickness=3).encode(
    x=alt.X("date:T"),
    y=alt.Y('close:Q'),
    color=alt.ColorValue('#a88f74'),
    opacity=alt.value(1.0),  # Always fully opaque when visible
    tooltip=[alt.Tooltip(district, type='nominal') for district in districts]
)

# Chart without the tooltip
bb_chart_without_tooltip = alt.Chart(df_bb).mark_tick(orient='vertical', thickness=3).encode(
    x=alt.X("date:T"),
    y=alt.Y('close:Q'),
    color=alt.ColorValue('#a88f74'),
    opacity=alt.value(1.0)  # Always fully opaque when visible
)

bb_chart = alt.layer(
    bb_chart_with_tooltip.add_params(bb_checkbox).transform_filter(bb_checkbox),
    bb_chart_without_tooltip.transform_filter(alt.datum.bb_checkbox == False)
)

# FOMC Dates

p = "Data/fomc_impact.csv"
df_fomc = pd.read_csv(p)
df_fomc = _to_datetime_filter(df_fomc, filter_=GE_1970)
df_fomc = df_fomc.merge(df_sp500[['date', 'close']], how='left', on='date')

bind_checkbox = alt.binding_checkbox(name='Add FOMC Date Ticks (green, red): ')
fomc_checkbox = alt.param(bind=bind_checkbox)

fomc_chart = alt.Chart(df_fomc).mark_tick(orient='vertical', thickness=3).encode(
    x=alt.X("date:T"),
    y=alt.Y('close:Q'),
    color=alt.condition(alt.datum.diff_norm > 0, alt.ColorValue('#64ff64'), alt.ColorValue('#ff6464')),
    opacity=alt.condition(fomc_checkbox, alt.value(1.), alt.value(0.)),
    tooltip=alt.condition(fomc_checkbox, "diff_norm:Q", alt.value(None))
).add_params(
    fomc_checkbox
)
#
chart = sp500_chart + fomc_chart + bb_chart
chart.save("Data/Viz/sp500_fomc_bb_viz.html")
