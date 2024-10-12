import json
import os
import tkinter as tk
from tkinter import Toplevel, messagebox

from PIL import Image, ImageTk  # PIL 모듈 추가


# 이미지 라벨링 프로그램 클래스
class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title('이미지 라벨링 프로그램')
        self.root.geometry('800x600')  # 창 크기를 800x600으로 설정

        # 이미지 폴더 경로
        self.image_folder = './data'
        self.images = [
            img
            for img in os.listdir(self.image_folder)
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        ]
        self.current_image_index = 0

        # JSON 데이터를 로드
        self.json_path = './labels.json'

        if not os.path.exists(self.json_path):
            # 파일 생성 (빈 파일 만들기)
            with open(self.json_path, 'w') as file:
                json.dump({}, file, indent=4)

        with open(self.json_path, 'r') as file:
            self.selections = json.load(file)

        # 이미지 레이블
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)

        # Canvas와 Scrollbar 사용하여 스크롤 가능한 버튼 영역 만들기
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.root, orient='vertical', command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))

        # 스크롤 가능한 프레임 생성
        self.button_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.button_frame, anchor='nw')

        # 1~95번 버튼 생성
        self.buttons = []
        buttons_per_row = 20  # 한 줄에 표시할 버튼 수 (20개로 변경)
        for i in range(1, 96):
            row = (i - 1) // buttons_per_row  # 행 번호
            col = (i - 1) % buttons_per_row  # 열 번호
            button = tk.Button(self.button_frame, text=str(i), command=lambda i=i: self.select_number(i))
            button.grid(row=row, column=col, padx=5, pady=5)
            self.buttons.append(button)

        # 제출 버튼
        self.submit_button = tk.Button(self.root, text='제출', command=self.submit_selection)
        self.submit_button.pack(pady=20)

        while self.images[self.current_image_index] in self.selections:
            self.current_image_index += 1

        self.open_new_window()
        self.show_image()

    def show_image(self):
        if self.current_image_index < len(self.images):
            image_path = os.path.join(self.image_folder, self.images[self.current_image_index])
            img = Image.open(image_path)

            # 기본값 설정 (창 크기를 가져오기 전에 사용할 값)
            max_width = self.root.winfo_width() if self.root.winfo_width() > 1 else 800  # 기본값 800
            max_height = self.root.winfo_height() if self.root.winfo_height() > 1 else 600  # 기본값 600

            # 창 크기 대비 이미지 크기를 조정
            max_width -= 40  # 창 너비보다 약간 작게 설정
            max_height -= 300  # 창 높이보다 약간 작게 설정

            # 이미지 크기가 창 크기를 넘으면 비율을 맞춰 조정
            img_width, img_height = img.size
            scale = min(max_width / img_width, max_height / img_height, 1)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)

            # Tkinter에서 사용할 수 있는 이미지 객체로 변환
            self.image = ImageTk.PhotoImage(img_resized)
            self.image_label.config(image=self.image)
            self.image_label.image = self.image  # 참조 유지

            # 선택된 버튼 색상 초기화
            self.update_button_colors()

            # 새로운 이미지를 표시할 때 선택 리스트 초기화
            if self.images[self.current_image_index] not in self.selections:
                self.selections[self.images[self.current_image_index]] = []
        else:
            messagebox.showinfo('완료', '모든 이미지 라벨링이 완료되었습니다!')
            self.root.quit()

    def select_number(self, number):
        current_image = self.images[self.current_image_index]
        # 선택한 번호가 리스트에 없으면 추가, 있으면 제거
        if number not in self.selections[current_image]:
            self.selections[current_image].append(number)
            self.buttons[number - 1].config(bg='lightgreen')  # 선택된 버튼 색상 변경
        else:
            self.selections[current_image].remove(number)
            self.buttons[number - 1].config(bg='SystemButtonFace')  # 기본 색상으로 복원

    def update_button_colors(self):
        current_image = self.images[self.current_image_index]
        selected = self.selections.get(current_image, [])

        for i, button in enumerate(self.buttons):
            if (i + 1) in selected:
                button.config(bg='lightgreen')  # 선택된 버튼은 초록색
            else:
                button.config(bg='SystemButtonFace')  # 선택되지 않은 버튼은 기본 색상

    def submit_selection(self):
        self.save_to_json()

        while self.images[self.current_image_index] in self.selections:
            self.current_image_index += 1

        if self.current_image_index < len(self.images):
            self.show_image()

    def save_to_json(self):
        json_file_path = './labels.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(self.selections, json_file, ensure_ascii=False, indent=4)

    def open_new_window(self):
        # 새 창 생성
        new_window = Toplevel(self.root)
        new_window.title('Carelabel Table')
        new_window.geometry('800x800')  # 이미지에 맞는 적절한 크기로 조정

        # 이미지 불러오기
        img = Image.open('carelabel_table.png')  # 파일 경로를 적절히 설정
        img = img.resize((800, 800))  # 이미지 크기 조정 (원하는 크기로 변경 가능)
        img_tk = ImageTk.PhotoImage(img)

        # 이미지 라벨 생성
        label = tk.Label(new_window, image=img_tk)
        label.image = img_tk  # 이미지가 사라지지 않도록 참조를 유지해야 함
        label.pack(pady=10)


# 메인 루프
if __name__ == '__main__':
    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.mainloop()
