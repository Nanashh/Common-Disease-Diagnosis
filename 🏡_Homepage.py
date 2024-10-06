from config import *

def main():
    st.title('Common Disease Symptoms Diagnosis')
    st.image("Images/pexels-shkrabaanthony-5214992.jpg")
    st.write("This Common Disease Symptom Diagnosis Web App is designed to provide users with quick and accessible insights into their potential health conditions based on symptoms they are experiencing. This web application leverages modern machine learning algorithms and a robust medical symptom database to assist users in identifying possible diseases related to their symptoms. The app is meant for educational and informational purposes, helping users better understand their health and prompting them to take appropriate actions, such as consulting a healthcare provider.")
    
    if st.button('Lets Start!', use_container_width=True, type='primary'):
        st.switch_page("pages/1_📊_Data_Exploration.py")

if __name__=='__main__':
    config= page_config('Homepage')
    main()