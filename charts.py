import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt

#histogram
rand=np.random.normal(1, 2, size=20)
fig, ax = plt.subplots()
ax.hist(rand, bins=15,color="purple")
st.pyplot(fig)



#line chart
df= pd.DataFrame(
    np.random.randn(10, 2)
    ,columns=['x', 'y'])
st.line_chart(df, color= ['#845EC2', '#D65DB1'])



#bar chart
st.bar_chart(df, color= ['#845EC2', '#D65DB1'])



#area chart
st.area_chart(df, color= ['#845EC2', '#D65DB1'])




#altair
df = pd.DataFrame(np.random.randn(500, 3),columns=['x','y','z'])
c = alt.Chart(df).mark_circle().encode(x='x',y='y' , size='z', color='z', tooltip=['x', 'y', 'z'])
st.altair_chart(c, use_container_width=True)



#diagram?
import streamlit as st
import graphviz as graphviz
st.graphviz_chart('''
                  digraph {
                          Big_shark -> Tuna 
                          Tuna -> Mackerel    
                          Mackerel -> Small_fishes     
                          Small_fishes -> Shrimp    
                  }'''
                  )


#maps
# df = pd.DataFrame(np.random.randn(500, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
# df = pd.DataFrame(np.random.randn(100, 2) / [1, 1] + [10, 117],columns=['lat', 'lon'])
df = pd.DataFrame(np.random.randn(100, 2) / [50, 50] + [14.5, 121],columns=['lat', 'lon'])

st.map(df)


st.balloons() #displays a fun animation of balloons on the screen.