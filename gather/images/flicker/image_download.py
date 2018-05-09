from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from time import sleep
import os
import random
from helper import *
from multiprocessing import Pool
import multiprocessing as mp
import concurrent.futures

MAX_WORKER = 10

def search_image(keyword, page):
  flickr = FlickrAPI(key, secret, format='parsed-json')
  res = flickr.photos.search(
    text = keyword,
    per_page = image_count,
    media = 'photos',
    sort = "relevance",
    safe_search = 1,
    extras = 'url_q,license',
    page = page
  )
  return res

def download_image(photos, savedir):
  try:
    for i, photo in enumerate(photos['photo']):
      url_q = photo['url_q']

      for ext in ['jpg', 'jpeg', 'png']:
        filepath = savedir+'/'+photo['id']+'.'+ext
        if os.path.exists(filepath):
          break

      print("file",str( i+1 ), "= ", url_q)
      urlretrieve(url_q, filepath)
      resize_bilinear(filepath, 28)
  
  except:
    import traceback
    traceback.print_exc()

# download images from Flickr
def main():
    for keyword in keywords:
        print("download now ...",  keyword)

        # create a directory to save images
        traindir = dataset_dir + keyword
        if not os.path.exists(traindir):
          os.mkdir(traindir)

        # search images
        for x in range(1, 20):
            res = search_image(keyword, x)
            photos = res['photos']

            # download images
            download_image(photos, traindir)
            sleep(random.randint(1,3))
     
if __name__ == '__main__':
    p = Pool(mp.cpu_count())
    executor = concurrent.futures.ThreadPoolExecutor(max_workers = MAX_WORKER)
    p.map(executor.submit(main()))
    p.close()