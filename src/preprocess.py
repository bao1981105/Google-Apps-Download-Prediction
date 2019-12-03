def preprocess(df,df2):
    import pandas as pd
    import numpy as np
    import re
    from datetime import datetime
    from sklearn.preprocessing import OrdinalEncoder
    # Make column 'Size' has the same unit of measurement: MB
    df['size_num'] = df.Size.str.extract('(\d+)') #.astype(int)
    df['size_letter'] = df.Size.str.extract('([a-zA-Z]\w{0,})')
    df['size_num'] = df[['size_num']].fillna(0)
    df['size_num'] = df['size_num'].astype(int)
    df.loc[df['size_letter'] == 'k',['size_num']] = df[df['size_letter']=='k']['size_num']/1000
    df.loc[df['Size']!='Varies with device',['Size']] = df['size_num']
    df.drop(['size_num','size_letter'],axis=1,inplace=True)
    df.drop_duplicates(keep = 'last', subset = ['App','Rating','Size','Type','Price','Content Rating', 'Genres'],inplace=True)
    
    # only a few data points missing
    df.dropna(subset=['Android Ver','Current Ver','Type'],inplace=True)
    df.reset_index(inplace=True,drop=True)
    
    # Split entries in column Genres
    df['Genres'] = df.Genres.str.split(';')

    genres = ["Action","Action & Adventure","Adventure","Art & Design","Arcade","Auto & Vehicles","Beauty","Board","Books & Reference","Brain Games","Business","Card","Casino","Casual","Comics","Communication","Creativity","Dating","Education","Educational","Entertainment","Events","Finance","Food & Drink","Health & Fitness","House & Home","Libraries & Demo","Lifestyle","Maps & Navigation","Medical","Music","Music & Audio","Music & Video","News & Magazines","Parenting","Personalization","Photography","Pretend Play","Productivity","Puzzle","Racing","Role Playing","Shopping","Simulation","Social","Sports","Strategy","Tools","Travel & Local","Trivia","Video Players & Editors","Weather","Word"]
    
    df_preprocessed = df
    df_preprocessed['Num_of_Characters'] = df_preprocessed['App'].str.len()
    def generate_lst(genre):
        lst_temp = []
        for i in range(len(df_preprocessed)):
            if genre in df_preprocessed.Genres[i]:
                lst_temp.append(1)
            else:
                lst_temp.append(0)
        return lst_temp
    d = {genre: generate_lst(genre) for genre in genres}
    transfored_genres = pd.DataFrame(d)
    df_preprocessed = pd.concat([df_preprocessed.drop('Genres',axis = 1),transfored_genres],axis=1)
    
    # Preprocess column Size
    df_preprocessed['Size_varies'] = df_preprocessed['Size'].apply(lambda x: 1 if x=='Varies with device' else 0)
    df_preprocessed['Size'] = df_preprocessed['Size'].replace(to_replace ='Varies with device', value = np.nan)
    
    # Preprocess target variable
    df_preprocessed['Installs'] = df_preprocessed['Installs'].replace(to_replace =['0+','0','1+','5+','10+','50+','100+','500+','1,000+','5,000+'], value = '<10k')
    df_preprocessed['Installs'] = df_preprocessed['Installs'].replace(to_replace =['10,000+','50,000+','100,000+'], value = '<500k')
    df_preprocessed['Installs'] = df_preprocessed['Installs'].replace(to_replace =['500,000+','1,000,000+'], value = '<5m')
    df_preprocessed['Installs'] = df_preprocessed['Installs'].replace(to_replace =['5,000,000+','10,000,000+','50,000,000+','100,000,000+','500,000,000+','1,000,000,000+'], value = '>5m')
    enc = OrdinalEncoder(categories = [['<10k','<500k','<5m','>5m']])
    df_preprocessed['Installs'] = enc.fit_transform(df_preprocessed[['Installs']])
    
    # Extract the most significant digit of Version number
    df_preprocessed['Current_Ver_truncated'] = df_preprocessed['Current Ver'].apply(lambda x: x.split('.')[0])
  
    df_preprocessed['Android_Ver_truncated'] = df_preprocessed['Android Ver'].apply(lambda x: x.split('.')[0])

    df_preprocessed.drop(['Current Ver','Android Ver'],axis = 1, inplace=True)
    
    df2_temp = df2[['App','Sentiment_Polarity','Sentiment_Subjectivity']].groupby(['App']).mean().reset_index()
    df_preprocessed = pd.merge(df_preprocessed,df2_temp,on='App',how='left')
   
    # preprocess Price column
    df_preprocessed.Price = df_preprocessed['Price'].apply(lambda x: float(re.findall(r'[\d\.\d]+', x)[0]))
    
    # preprocess Last Updated column
    df_preprocessed['Last Updated'] = df_preprocessed['Last Updated'].apply(lambda x: datetime.strptime(x, '%d-%b-%y'))
    df_preprocessed['Last Updated'] = df_preprocessed['Last Updated'].astype(np.int64)
    
    # Since 91.5% of values are missing, we drop these two columns
    df_preprocessed.drop(['Sentiment_Polarity','Sentiment_Subjectivity'],axis=1,inplace=True)
    
    # drop App name
    df_preprocessed.drop(['App'],axis=1,inplace=True)
    return df_preprocessed


