import sys
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("Plot")

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.lineEdit = QLineEdit()  # 품목번호 입력
        self.pushButton = QPushButton("Plot")  # Plot 버튼
        self.pushButton.clicked.connect(self.pushButtonClicked)

        # 나타낼 항목 선택지
        self.comboBox = QComboBox(self)
        self.comboBox.addItems(['단가(원)', '거래량', '거래대금(원)', '경매건수', '해당일자_전체평균가격(원)', '해당일자_전체거래물량(kg)',
                                '하위가격 평균가(원)', '상위가격 평균가(원)', '하위가격 거래물량(kg)', '상위가격 거래물량(kg)',
                                '일자별_도매가격_최대(원)', '일자별_도매가격_평균(원)', '일자별_도매가격_최소(원)',
                                '일자별_소매가격_최대(원)', '일자별_소매가격_평균(원)', '일자별_소매가격_최소(원)', '수출중량(kg)',
                                '수출금액(달러)', '수입중량(kg)', '수입금액(달러)', '무역수지(달러)', '주산지_0_초기온도(℃)',
                                '주산지_0_최대온도(℃)', '주산지_0_최저온도(℃)', '주산지_0_평균온도(℃)', '주산지_0_강수량(ml)',
                                '주산지_0_습도(%)', '주산지_1_초기온도(℃)', '주산지_1_최대온도(℃)', '주산지_1_최저온도(℃)',
                                '주산지_1_평균온도(℃)', '주산지_1_강수량(ml)', '주산지_1_습도(%)', '주산지_2_초기온도(℃)',
                                '주산지_2_최대온도(℃)', '주산지_2_최저온도(℃)', '주산지_2_평균온도(℃)', '주산지_2_강수량(ml)', '주산지_2_습도(%)'])
        self.comboBox.move(50, 50)

        # Left Layout
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)

        # Right Layout
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.lineEdit)
        rightLayout.addWidget(self.comboBox)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)
        self.setLayout(layout)

    def pushButtonClicked(self):
        num = self.lineEdit.text()
        leg = self.comboBox.currentText()

        ddf = pd.read_csv(f'./data/train/train_{num}.csv')
        datadate = []
        for i, date in enumerate(ddf['datadate']):
            datadate.append(datetime.strptime(str(date), "%Y%m%d"))  # 날짜를 string에서 날짜로 바꿈

        plt.rc('font', family="NanumGothic")  # 한글 폰트 설정
        self.ax.clear()  # 그려져 있는 플롯 삭제
        self.ax.plot(datadate, ddf[leg], label=leg)
        self.ax.set_title(f'품목 {num}')  # 제목
        self.ax.legend(loc='upper left')  # 범례
        self.canvas.draw()  # 최종 plot


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
