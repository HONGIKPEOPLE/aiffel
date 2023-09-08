import numpy as np
import cv2 as cv

# 피부색 마스크 생성 함수
def skinmask(img):
    # 이미지를 HSV 색상 공간으로 변환
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # 피부색 범위 정의
    lower = np.array([0, 40, 15], dtype="uint8")   # 순서대로 색상(h),채도(s),명도(v)   채도 = 진함의 정도/ 명도= 밝음의 정도
    upper = np.array([35, 255, 255], dtype="uint8")
    
    # 피부색 범위 내의 픽셀을 마스크로 선택
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    '''inrange 함수를 통해서 이미지/영상 출력해보기'''

    # 마스크를 블러 처리하여 노이즈 제거
    blurred = cv.blur(skinRegionHSV, (10, 10))  # (10,10)는 커널의 크기 - 블러 처리의 강도와 방향을 결정   
    '''커널이 뭔지 찾기,어떻게 동작하는 건지 + 블러가 어떤 방식으로 되는지  이미지/영상'''
    
    # 이진화 처리하여 피부색 픽셀을 강조
    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)  # 0은 임계값, 255는 임계값을 초과하는 경우에 설정할 픽셀 값(일반적으로 흰색)
    ''' thresh 띄워보기 이미지/영상  0,255는 무슨값?? 채널은 RGB인데 왜 숫자가 1개??'''
    
    return thresh

# 손의 윤곽과 convex hull 얻는 함수
def getcnthull(mask_img):
    # 마스크 이미지로부터 윤곽(contours)을 찾음
    # cv.RETR_TREE: 윤곽선 검출 방법을 지정하는 매개변수로, cv.RETR_TREE는 윤곽선들의 계층 구조 정보를 반환
    # cv.CHAIN_APPROX_SIMPLE: 윤곽선의 근사 방법을 지정하는 매개변수로, cv.CHAIN_APPROX_SIMPLE는 윤곽선의 꼭지점의 좌표만 반환/저장
    contours, hierarch = cv.findContours(mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 윤곽(contours)을 선택하여 손의 윤곽으로 사용
    # key=lambda x: cv.contourArea(x): max 함수의 key 매개변수를 사용하여 윤곽선을 비교하는 방법을 지정합니다. cv.contourArea(x)는 윤곽선 x의 면적을 계산하는 함수를 호출하는 람다 함수
    contours = max(contours, key=lambda x: cv.contourArea(x))
    
    # 손의 윤곽(contours)에 대한 convex hull을 계산 //  cv.CHAIN_APPROX_SIMPLE는 윤곽선의 꼭지점의 좌표만 반환/저장 -- 이걸로 그림
    hull = cv.convexHull(contours)
    
    return contours, hull

# 손의 볼록성 결함(defects) 얻는 함수
def getdefects(contours):
    # 손의 윤곽(contours)에 대한 convex hull 계산 (returnPoints=False로 설정)
    # returnPoints=False로 설정하면, 함수는 볼록 껍질의 점 인덱스를 나타내는 리스트를 반환합니다.
    hull = cv.convexHull(contours, returnPoints=False)  
    ''' 무슨 리스트를 반환하는 건지 검색'''
    
    # 볼록성 결함(defects) 찾기 // 최대 편차를 갖는 곳(?)
    defects = cv.convexityDefects(contours, hull)
    
    
    return defects  
    ''' print de 해보기 '''

cap = cv.VideoCapture(0)  # 웹캠 열기 ('0'는 웹캠 인덱스)

while cap.isOpened():
    _, img = cap.read()  # 웹캠에서 프레임 읽기
    img = cv.flip(img,1)
    
    try:
        mask_img = skinmask(img)  # 피부색 마스크 생성
        contours, hull = getcnthull(mask_img)  # 손의 윤곽과 convex hull 얻기 
        cv.drawContours(img, [contours], -1, (255, 255, 0), 2)  # 손의 윤곽 그리기// (255, 255, 0), 2 -- 색상과 두께 BGR
        cv.drawContours(img, [hull], -1, (0, 255, 255), 2)  # convex hull 그리기// (0, 255, 255), 2 -- 색상과 두께  BGR
        defects = getdefects(contours)  # 볼록성 결함(defects) 얻기

        # 손가락 갯수 세기
        if defects is not None and len(contours) > 160:  # 320은 임의의 임계값 // 환경에 따라 직접 수정해야 할 파라미터
            cnt = 0
            for i in range(defects.shape[0]):
                # s = defect의 시작점 / e = 끝 점/ f = 꼭지점 / d = 거리
                ''' s,e,f,d 가 defect의 어느 지점을 말하는 건지 defect가 점은 맞는지? '''
                s, e, f, _ = defects[i][0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))     # 이 계산방식은 구글에 매우 많음 // https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08
                if angle <= np.pi / 2:  # 90도 이하 각도면 손가락으로 인식
                    cnt += 1
                    cv.circle(img, far, 4, [0, 0, 255], -1)   # (원을 그릴 이미지,원의 중심 좌표를 나타내는 튜플 (x, y),반지름,색상,-1은 원 내부를 채운다 )
                    ''' far가 왜 원의 중심인지 sefd를 공부하면 자연스럽게 알 수 있음'''
            if cnt >= 0:
                cnt += 1
                if cnt >= 5:  # 5개 이상 손가락인 경우 5로 제한
                    cnt = 5                    
            cv.putText(img, str(len(contours)), (0, 22), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)  # contour 길이
            cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)     # (0,50) = 위치/ '1' = 크기 /(255, 0, 0): 텍스트의 색상을 나타내는 튜플 /'2' = 굵기 / cv.LINE_AA = 랜더링??
        elif len(contours) <= 160:  # 임계값 이하인 경우 손가락 없음 // 환경에 따라 직접 수정해야 할 파라미터
            cnt = 0
            cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
            cv.putText(img, str(len(contours)), (0, 22), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        
        # 영상 출력
        cv.imshow("img", img)
    except:
        pass
    
    # 'q' 키를 누르면 루프 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 웹캠 해제
cv.destroyAllWindows()  # 모든 OpenCV 창 닫기
