from config import *

def word_prep(df):
    punct=string.punctuation
    english_stopwords= stopwords.words('english')
    tokens = word_tokenize(df.lower())
    tokens = [token.lower() for token in tokens if token.isalpha() and token not in english_stopwords and token not in punct]
    return ' '.join(tokens)

def trainmodel(xtr,xte,ytr,yte, params):
    rfc=RandomForestClassifier(n_estimators=params[0],max_depth=params[1],criterion=params[2],class_weight=params[3])
    rfc.fit(xtr,ytr)
    yp=rfc.predict(xte)
    return rfc, yp, yte

def evaluate(yp,yte):
    acc=accuracy_score(yte,yp)
    st.header(f'accuracy = {acc}')
    st.header('Confusion Matrix') 
    cm=confusion_matrix(yte,yp)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm,annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)

def splitting(x,y,params):
   x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
   return trainmodel(x_train,x_test,y_train,y_test,params)

def savemodel(model,path):
    with open(path, 'wb') as file:
            pickle.dump(model, file)

def training(x,y, params):
    model,yp,yte=splitting(x,y,params)
    savemodel(model,f'model/{params[4]}.pickle')
    return yp,yte

def preprocessing(data):
    data['Disease']=data['Disease'].apply(word_prep)
    tfidf=TfidfVectorizer()
    x=tfidf.fit_transform(data['text'])
    le=LabelEncoder()
    y=le.fit_transform(data['Disease'])
    st.session_state['tfidf']=tfidf
    st.session_state['le']=le
    return x,y

def main():
    st.title('Train Model')
    params=[100,20,'gini','balanced','default']
    #pred,tes=training(data,params)
    #evaluate(pred,tes)

    
    if 'data' not in st.session_state:
        if st.button('Explore Data First!', type='primary'):
            st.switch_page("pages/1_📊_Data Exploration.py")
    else:
        st.header('Choose your model :black_nib:', divider='grey')
        st.info(f'''Default model: **Random Forest Classifier**.''', icon='ℹ')
        active=st.toggle('Train your own model')
        data=st.session_state['data']
        x,y=preprocessing(data)

        if active:
            with st.form('model training'):
                n_est = st.number_input('Enter the number of estimators: ', min_value=0, value=100)
                m_dep = st.number_input('Enter the max depth: ', min_value=0, value=20)
                crit = st.selectbox('Enter the criterion',  ["gini", "entropy","log_loss"], index=0)
                c_wght = st.selectbox('Enter the class weight', ["balanced", "balanced_subsample"], index=0)
                model_name = st.text_input("Enter your model name", value='user_model')
                if st.form_submit_button("Submit"):
                    params=[n_est,m_dep,crit,c_wght,model_name]
                    pred,tes=training(x,y,params)
                    evaluate(pred,tes)
                    st.info('model trained sucessfully!')
        
        if st.button('Predict Disease', type='primary'):
            st.switch_page("pages/3_🩺_Disease.py")
          
if __name__=='__main__':
    config= page_config('Predict')
    main()