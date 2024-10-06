from config import *

def load_data():
    data1=pd.read_csv('Dataset/dataset(5).csv')
    data2=pd.read_csv('Dataset/Symptom2Disease(2).csv')
    data2.drop(['Unnamed: 0'], axis=1,inplace=True)
    return data1,data2

def description():
    st.info(f'''2 common disease symptoms datasets are used to broaden data coverage''', icon='ℹ️')

def joinsymptoms(text):
    return ','.join(text.dropna().astype(str))

def joinsymp(df1):
    df1['text']=df1[df1.columns[1:]].apply(joinsymptoms, axis=1)
    delsymp=df1.drop(['Disease','text'],axis=1)
    df1=df1.drop(delsymp,axis=1)
    return df1

def lower(text):
  tokens=text.lower()
  return tokens

def concat(df1,df2):
    df2.columns=(['Disease','text'])
    df=pd.concat([df1,df2], ignore_index=True)
    df['Disease']=df['Disease'].apply(lower)
    return df

def pie(df):
    disease_counts = df['Disease'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(disease_counts, labels=disease_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    return fig

def mostword(df):
    wc=WordCloud(width=1000, height=500,min_font_size=12)
    wc.generate(''.join(df['text']))

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

def resampl(df1):
    df_balanced = pd.DataFrame()  # Empty DataFrame to store results
    for class_value in df1['Disease'].unique():
        df_class = df1[df1['Disease'] == class_value]
        if len(df_class) < df1['Disease'].value_counts().max():
            df_class_resampled = resample(df_class,
                                        replace=True,
                                        n_samples=df1['Disease'].value_counts().max(),
                                        random_state=42)
        else:
            df_class_resampled = df_class
        df_balanced = pd.concat([df_balanced, df_class_resampled])

    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
    return df_balanced

def main():
    st.title("Exploratory Data Analysis")
    description()
    df1,df2=load_data()
    tab1,tab2,tab3,tab4=st.tabs(["Datasets","Joined Data","Disease Available","Most Common Symptoms"])
    with tab1:
        data1,data2=st.columns(2)

        with data1:
            st.subheader('Dataset 1')
            st.write('Disease and symptoms dataset with 41 different diseases')
            st.dataframe(df1)
        with data2:
            st.subheader('Dataset 2')
            st.write('Disease and symptoms dataset with 24 different diseases')
            st.dataframe(df2)

    with tab2:
        st.subheader('Joined Data')
        st.write('Joined disease and symptoms dataset with 46 different diseases')
        df1=joinsymp(df1)
        df=concat(df1,df2)
        st.dataframe(df)

    with tab3:
        st.subheader('Diseases Available')
        st.write('Disease coverage and distribution')
        #st.dataframe(df['Disease'].value_counts().to_frame().T)
        df=resampl(df)
        st.pyplot(pie(df))

    with tab4:
        st.subheader('Most common symptoms')
        st.write('Symptoms which appears commonly in diseases. Bigger the font size, means more common the symptoms')
        st.pyplot(mostword(df))

    st.session_state['data']=df

    if st.button("Predict your disease 👉", type='primary'):
        st.switch_page("pages/2_🔍_Predict.py")

    

if __name__=='__main__':
    config= page_config('Data Exploration')
    main()