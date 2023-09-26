import streamlit as st
import time

#headers
# st.write("Hello ,let's learn how to build a streamlit app together")
# st.title ("this is the app title")
# st.header("this is the markdown")
# st.markdown("this is the header")
# st.subheader("this is the subheader")
# st.caption("this is the caption")
# st.code("x=2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')


#images, audio and video
st.image("Lit.jpg")
#st.audio("Audio.mp3")
#st.video("video.mp4")


#input
# st.checkbox('yes')
# st.button('Click')
# st.radio('Pick your gender',['Male','Female'])
# st.selectbox('Pick your gender',['Male','Female'])
# st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
# st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
# st.slider('Pick a number', 0,50)

# CTRL / for comment 


#animation
# st.balloons() #displays a fun animation of balloons on the screen.
# st.progress(100) #creates a progress bar that is 10% complete.
# with st.spinner('Wait for it...'):     #creates a spinner animation with the message "Wait for it..." and then pauses the program for 10 seconds using the time.sleep() function.
#     time.sleep(10) 


# st.success("You did it !")
# st.error("Error")
# st.warning("Warning")
# st.info("It's easy to build a streamlit app")
# st.exception(RuntimeError("RuntimeError exception"))

st.sidebar.title("Sidebar!")

st.sidebar.button("Button!")


st.sidebar.radio('Pick your gender',['Male','Female'])


# But st.spinner() and st.echo() are not supported with st.sidebar.


container = st.container()
container.write("This is written inside a container!")

container.write("Still inside a container!")
st.write("This is written OUTSIDE a container!")

