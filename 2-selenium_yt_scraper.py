from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import re
import pandas as pd 
from tqdm import tqdm


options = webdriver.ChromeOptions() 
options.add_argument("start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path='chromedriver.exe')
baseurl = "http://youtube.com"
data_basics = pd.read_csv('final_result.csv')

yt = []
try:
    for index, row in tqdm(data_basics.iterrows(), total=data_basics.shape[0]):
        keyword = row['originalTitle']+ ' ' + str(int(row['startYear']))+ ' trailer' #'barbara trailer'#input()
        driver.get(f'{baseurl}/search?q={keyword}')
        print("****************")
        print(row['originalTitle'])
        WebDriverWait(driver, 20).until(EC.visibility_of_all_elements_located((By.XPATH, "//yt-formatted-string[@class='style-scope ytd-video-renderer' and @aria-label]")))[0].click()
        try:
            viewsNum = WebDriverWait(driver, 3).until(EC.visibility_of_all_elements_located((By.XPATH, "//span[@class='view-count style-scope yt-view-count-renderer']")))[0].text
            viewsNum = int(re.sub('\D', '', viewsNum))
            print("VIEWS: ", str(viewsNum))
        except:
            viewsNum = 'NA'
            print("VIEWS: ", str(viewsNum))
        driver.execute_script("window.scrollTo(0, 200)")
        try:
            commentCount = WebDriverWait(driver, 4).until(EC.visibility_of_all_elements_located((By.XPATH, "//yt-formatted-string[@class='count-text style-scope ytd-comments-header-renderer']")))[0].text
            commentCount = int(re.sub('\D', '', commentCount))
            print("Comment Count: ", str(commentCount))
        except:
            commentCount = 'NA'
            print("Comment Count: ", str(commentCount))
        try:
            likeCount = WebDriverWait(driver, 3).until(EC.visibility_of_all_elements_located((By.XPATH, "//yt-formatted-string[@class='style-scope ytd-toggle-button-renderer style-text' and contains(@aria-label, 'gostaram')]")))[0].get_attribute("aria-label")
            likeCount = int(re.sub('\D', '', likeCount))
            print("likeCount: ", str(likeCount))
        except:
            likeCount = 'NA'
            print("likeCount: ", str(likeCount))
        try:
            dislikeCount = WebDriverWait(driver, 3).until(EC.visibility_of_all_elements_located((By.XPATH, "//yt-formatted-string[@class='style-scope ytd-toggle-button-renderer style-text' and contains(@aria-label, 'marcações \"Não gostei\"')]")))[0].get_attribute("aria-label")
            dislikeCount = int(re.sub('\D', '', dislikeCount))
            print("dislikeCount: ", str(dislikeCount))
        except:
            dislikeCount = 'NA'
            print("dislikeCount: ", str(dislikeCount))
        print("****************")
        yt.append(str(viewsNum)+';'+str(likeCount)+';'+str(dislikeCount)+';'+str(commentCount)+';')
except:
    pass

data_basics["yt_stats"] = yt
data_basics["viewCount"] = data_basics.apply(lambda x: x["yt_stats"].split(";")[0],axis=1)
data_basics["likeCount"] = data_basics.apply(lambda x: x["yt_stats"].split(";")[1],axis=1)
data_basics["dislikeCount"] = data_basics.apply(lambda x: x["yt_stats"].split(";")[2],axis=1)
data_basics["commentCount"] = data_basics.apply(lambda x: x["yt_stats"].split(";")[3],axis=1)
data_basics.drop("yt_stats", axis=1, inplace=True)
print(data_basics.head(5))
data_basics.to_csv("final_result_youtube_stats.csv")