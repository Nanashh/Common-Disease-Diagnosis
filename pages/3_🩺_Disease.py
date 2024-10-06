from config import *

def load_model():
    folder_path = 'model'
    file_list = os.listdir(folder_path)
    pickle_files = [f for f in file_list if f.endswith('.pickle')]
    
    model = {}
    for pickle_file in pickle_files:
        file_path = os.path.join(folder_path, pickle_file)
        with open(file_path, 'rb') as file:
            model[pickle_file] = pickle.load(file)
    return model

def model_selection():
    MODELS=load_model()
    model=MODELS[st.selectbox('Choose your model', MODELS.keys())]
    return model

def main():
    st.title('Disease Prediction')

    if 'data' not in st.session_state and 'tfidf' not in st.session_state and 'le' not in st.session_state:
        if st.button('Train model first!', type='primary'):
            st.switch_page("pages/2_🔍_Predict.py")
    else:
        with st.form('predict data'):
            st.header('Predict Disease', divider='grey')
            model=model_selection()
            text=st.text_input('Symptoms descriptions')
            if st.form_submit_button("Submit"):
                df=st.session_state['data']
                tf=st.session_state['tfidf']
                le=st.session_state['le']
                text=tf.transform([text])
                pred=model.predict(text)
                pred=le.inverse_transform(pred)
                st.subheader(f'Your disease might be {pred[0]}')

if __name__=='__main__':
    config= page_config('Disease')
    main()