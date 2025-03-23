import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 읽기
file_path = "./data.csv"  # CSV 파일 경로를 지정하세요.
data = pd.read_csv(file_path)

# 데이터 확인 (예: 레이어 이름과 각 구성 요소의 값)
print(data)

# 데이터 컬럼 정의 (CSV 파일에 따라 수정 필요)
layers = data['Layer']  # 레이어 이름 (예: CONV1, CONV2 등)
dram = data['DRAM']     # DRAM 값
buffer = data['Buffer'] # Buffer 값
array = data['Array']   # Array 값
rf = data['RF']         # RF 값

# x축 위치 설정
x = np.arange(len(layers))

# 그래프 크기 설정
plt.figure(figsize=(10, 5))

# 스택 막대 그래프 생성
plt.bar(x, dram, label='DRAM', color='gray')
plt.bar(x, buffer, bottom= dram, label='Buffer', color='darkgray')
plt.bar(x, array, bottom=np.add(dram, buffer), label='Array', color='lightgray')
plt.bar(x, rf, bottom=np.add(np.add(dram, buffer), array), label='RF', color='white', edgecolor='black')

# x축 및 y축 레이블 설정
plt.xticks(x, layers)
plt.ylabel('Normalized Energy')
plt.title('Energy Consumption by Layer and Component')

# 범례 추가
plt.legend()

# 그래프 출력
plt.show()
