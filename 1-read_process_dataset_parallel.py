import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import multiprocessing
import numpy as np

def youtube_request(title):
    try:
        print("*******************")
        print(title)
        print("*******************")
        query = title.replace(" ", "%20")
        key = 'AIzaSyDQA11UpgeWkpQtij-adF8RBCFVP67Fa9k'
        result = requests.get('https://www.googleapis.com/youtube/v3/search?part=snippet&q='+query+'&key='+key)
        videoId = result.json().get("items")[0].get("id").get("videoId")
        channelId = result.json().get("items")[0].get("snippet").get("channelId")
        statistics = requests.get('https://www.googleapis.com/youtube/v3/videos?part=statistics&id='+videoId+'&key='+key)
        print(statistics.json().get("items")[0].get("statistics"))
        time.sleep(1)
        return str(statistics.json().get("items")[0].get("statistics").get("viewCount"))+";"+str(statistics.json().get("items")[0].get("statistics").get("likeCount"))+";"+str(statistics.json().get("items")[0].get("statistics").get("dislikeCount"))+";"+str(statistics.json().get("items")[0].get("statistics").get("commentCount"))
    except:
        return "0;0;0;0"

def actors_exp(df):
    actorsExp = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        actorsOfMovie = data_actors[data_actors['tconst'] == row['tconst']]
        numberOfTitlesExperiencedByActors = 0
        for index2, row2 in actorsOfMovie.iterrows():
            persons = data_persons[data_persons['nconst'] == row2['nconst']]
            for index3, row3 in persons.iterrows():
                #print(row3)
                numberOfTitlesExperiencedByActors += len(row3['knownForTitles'].split(","))
        actorsExp.append(numberOfTitlesExperiencedByActors)
    df['staffExperience'] = actorsExp
    return df

def writers_dirs_exp(df):
    writerExp = []
    directorExp = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        directorsWritersOfMovie = data_dir_writer[data_dir_writer['tconst'] == row['tconst']]
        directorsExp = 0
        writersExp = 0
        for index2, row2 in directorsWritersOfMovie.iterrows():
            for director in row2['directors'].split(','):
                persons = data_persons[data_persons['nconst'] == director]
                for index3, row3 in persons.iterrows():
                    #print(row3)
                    directorsExp += len(row3['knownForTitles'].split(","))
            for writer in row2['writers'].split(','):
                persons = data_persons[data_persons['nconst'] == writer]
                for index3, row3 in persons.iterrows():
                    #print(row3)
                    writersExp += len(row3['knownForTitles'].split(","))
        writerExp.append(writersExp)
        directorExp.append(directorsExp)
    df['writersExperience'] = writerExp
    df['directorsExperience'] = directorExp
    return df

def budget_revenue(df):
    budgets=[]
    revenues=[]
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        budget=0
        revenue=0
        movies = tmdb_dataset[tmdb_dataset['original_title'] == row['originalTitle']]
        for index2, row2 in movies.iterrows():
            budget += row2['budget']
            revenue += row2['revenue']
        budgets.append(budget)
        revenues.append(revenue)
    df['budget'] = budgets
    df['revenue'] = revenues
    return df

def parallelize_dataframe(df, func):
    num_cores = multiprocessing.cpu_count()-2  #leave one free to not freeze machine
    num_partitions = num_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

tmdb_dataset = pd.read_csv('tmdb_movies_data.csv', sep=',')
data_basics = pd.read_csv('title.basics.tsv/data.tsv', sep='\t')
data_basics = data_basics[data_basics["titleType"].str.contains("movie")]
data_basics.drop("endYear", axis=1, inplace=True)
print(data_basics.head(50))
print(data_basics.info())
data_basics["startYear"] = pd.to_numeric(data_basics["startYear"],errors='coerce')
data_basics["runtimeMinutes"] = data_basics["runtimeMinutes"].apply (pd.to_numeric, errors='coerce')
data_basics = data_basics.dropna()
data_basics = data_basics[data_basics["genres"] != '\\N']
#data_basics = data_basics[data_basics["isAdult"] == 1]
data_basics = data_basics[~data_basics["genres"].str.contains('Documentary')]
data_basics = data_basics[data_basics["originalTitle"].isin(tmdb_dataset['original_title'].values)]
data_basics = data_basics[data_basics["startYear"] >= 2000]


print(data_basics.head(20))
#YOUTUBE FEATURES/STATISTICS



data_actors = pd.read_csv('title.principals.tsv/data.tsv', sep='\t')
data_dir_writer = pd.read_csv('title.crew.tsv/data.tsv', sep='\t')
data_persons = pd.read_csv('name.basics.tsv/data.tsv', sep='\t')
print(data_actors.head())
print(data_persons.head())

print("ITERATE OVER ROWS OF MOVIE:")



data_basics = parallelize_dataframe(data_basics, actors_exp)
data_basics = parallelize_dataframe(data_basics, writers_dirs_exp)
data_basics = parallelize_dataframe(data_basics, budget_revenue)


data_basics = data_basics[data_basics["revenue"] != 0]
data_basics = data_basics[data_basics["budget"] != 0]

print( "NUMBER OF ROWS DATAFRAME FINAL: ",len(data_basics.index))
#data_basics = data_basics.sample(n = 5)
'''
data_basics["yt_stats"] = data_basics.apply(lambda x: youtube_request(x["originalTitle"]),axis=1)
data_basics["viewCount"] = data_basics.apply(lambda x: x["yt_stats"].split(";")[0],axis=1)
data_basics["likeCount"] = data_basics.apply(lambda x: x["yt_stats"].split(";")[1],axis=1)
data_basics["dislikeCount"] = data_basics.apply(lambda x: x["yt_stats"].split(";")[2],axis=1)
data_basics["commentCount"] = data_basics.apply(lambda x: x["yt_stats"].split(";")[3],axis=1)
data_basics.drop("yt_stats", axis=1, inplace=True)
'''
print(data_basics.head(5))

data_basics.to_csv("final_result.csv")

#FOR EACH TITLE GET NCONST IN TITLE.PRINCIPALS AND SUM NUM OF KNOWN TITLES FROM NAMES.BASICS
