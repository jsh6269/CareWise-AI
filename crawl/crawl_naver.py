import os

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# 요청 헤더 설정
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
        Chrome/107.0.1418.62 Safari/537.36 Edg/107.0.1418.62'
}

# 저장할 폴더 경로 설정
save_folder = './data'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


for page_num in tqdm(range(1, 300)):
    hrefs = []

    # 요청할 URL 설정
    # 세탁 기호
    url = f'https://kin.naver.com/search/list.naver?query=\
        %EC%84%B8%ED%83%81%20%EA%B8%B0%ED%98%B8&section=qna&page={page_num}'

    # 세탁 라벨
    # url = f'https://kin.naver.com/search/list.naver?query=\
    # %EC%84%B8%ED%83%81%20%EB%9D%BC%EB%B2%A8&page={page_num}'

    # requests를 이용한 GET 요청
    response = requests.get(url, headers=headers)

    # 요청 성공 여부 확인
    if response.status_code == 200:
        # HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        # 모든 li 요소 찾기
        li_elements = soup.select('#s_content > div.section > ul > li')

        # 각 li 요소 탐색
        for li in li_elements:
            # li의 하위에 div 태그가 있는지 확인
            if li.find('div'):
                # li 하위의 dl > dt > a 요소 찾기
                a_tag = li.select_one('dl > dt > a')
                if a_tag and a_tag.get('href'):
                    hrefs.append(a_tag['href'])
    else:
        print('Failed to retrieve the webpage. Status code:', response.status_code)

    # hrefs에 있는 URL 순회
    for idx, href_url in enumerate(hrefs):
        response = requests.get(href_url, headers=headers)

        # 요청 성공 여부 확인
        if response.status_code == 200:
            # HTML 파싱
            soup = BeautifulSoup(response.text, 'html.parser')

            # 이미지 URL 추출
            img_tag = soup.select_one(
                '#content > div.endContentLeft._endContentLeft > div.contentArea._contentWrap\
                  > div.questionDetail ._waitingForReplaceImage'
            )

            # 이미지가 있는지 확인
            if img_tag and img_tag.get('data-image-src'):
                img_url = img_tag['data-image-src']
                filename = img_url.split('//')[-1].replace('/', '_')
                img_filename = os.path.join(save_folder, filename)

                if os.path.exists(img_filename):
                    print('exists')
                    pass
                else:
                    try:
                        # 이미지 요청
                        img_response = requests.get(img_url, headers=headers, timeout=10)
                    except requests.exceptions.Timeout:
                        print(
                            f'Timeout occurred while trying to download image from {img_url}. Continuing to next image.'
                        )
                        continue  # 다음 이미지로 넘어감

                    if img_response.status_code == 200:
                        # 파일 저장 경로 설정
                        with open(img_filename, 'wb') as f:
                            f.write(img_response.content)
                        # print(f'Saved image {img_filename}')
                    else:
                        print(f'Failed to download image. Status code: {img_response.status_code}')
            else:
                print('No image found in the specified selector.')
        else:
            print(f'Failed to retrieve the webpage for {href_url}. Status code:', response.status_code)
