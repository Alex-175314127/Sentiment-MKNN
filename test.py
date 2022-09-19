import altair as alt
import pandas as pd
import streamlit as st

df = pd.DataFrame(
    {
        'Duration': [f"{d} Days" for d in [120, 60, 30, 14, 7]],
        'Bradley Beal': [10,20,30,40,50],
        'Steven Adams': [30,10,50,90,70]
    },
    columns=['Duration', 'Bradley Beal', 'Steven Adams']
)

st.write(df)

st.markdown("Make a long form dataframe: https://altair-viz.github.io/user_guide/data.html#long-form-vs-wide-form-data")

df = df.melt('Duration', var_name='name', value_name='value')
st.write(df)

chart = alt.Chart(df).mark_line().encode(
  x=alt.X('Duration:N'),
  y=alt.Y('value:Q'),
  color=alt.Color("name:N")
).properties(title="Herld")
st.altair_chart(chart, use_container_width=True)