import cv2
import numpy as np
import random
from keras.models import load_model
import time
import pygame

# Khởi tạo pygame mixer
pygame.mixer.init()

# Tải nhạc nền và âm thanh hiệu ứng
background_music = pygame.mixer.music.load("path_to_your_background.mp3")
draw_sound = pygame.mixer.Sound("draw_sound.mp3")
win_sound = pygame.mixer.Sound("win_sound.mp3")
lose_sound = pygame.mixer.Sound("lose_sound.mp3")

# Phát nhạc nền (vòng lặp liên tục)
pygame.mixer.music.play(-1, 0.0)  # -1 có nghĩa là phát nhạc nền liên tục

# Vô hiệu hóa ký hiệu khoa học để dễ đọc kết quả hơn
np.set_printoptions(suppress=True)

# Tải mô hình đã được huấn luyện trước đó
model = load_model("keras_model.h5", compile=False)

# Tải các nhãn (Kéo, Bao, Búa)
class_names = open("labels.txt", "r").readlines()

# Kiểm tra xem lớp Undefined có tồn tại không
undefined_class = "Undefined"

# Khởi tạo camera
camera = cv2.VideoCapture(0)

# Thiết lập tổng số vòng chơi
total_rounds = 5
current_round = 1

# Điểm số của người chơi và AI
user_score = 0
ai_score = 0

# Các lựa chọn của AI dựa trên nhãn
ai_choices = [class_names[i][2:].strip() for i in range(len(class_names))]

def determine_winner(user_choice, ai_choice):
    """
    Xác định người thắng dựa trên lựa chọn của người chơi và AI.
    Quy tắc:
    - Búa thắng Kéo
    - Kéo thắng Bao
    - Bao thắng Búa
    """
    if user_choice == ai_choice:
        return "Draw"  # Hòa
    elif (user_choice == "Rock" and ai_choice == "Scissors") or \
         (user_choice == "Scissors" and ai_choice == "Paper") or \
         (user_choice == "Paper" and ai_choice == "Rock"):
        return "User Wins"  # Người chơi thắng
    else:
        return "AI Wins"  # AI thắng

def update_scores(result):
    global user_score, ai_score
    if result == "User Wins":
        user_score += 1
    elif result == "AI Wins":
        ai_score += 1

print("Trò chơi bắt đầu! Chơi Kéo, Bao, Búa với AI.")
print("Nhấn ESC để thoát trò chơi sớm.")

while current_round <= total_rounds:
    print(f"Vòng {current_round}/{total_rounds}: Sẵn sàng!")
    time.sleep(2)  # Tạm dừng ngắn trước khi bắt đầu

    # Chụp khung hình từ camera
    ret, frame = camera.read()
    if not ret:
        print("Không thể lấy khung hình. Thoát...")
        break

    # Thay đổi kích thước khung hình để phù hợp với đầu vào của mô hình
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Tiền xử lý khung hình để dự đoán
    image = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1  # Chuẩn hóa về khoảng [-1, 1]

    # Dự đoán
    prediction = model.predict(image)
    user_index = np.argmax(prediction)
    user_choice = class_names[user_index][2:].strip()
    confidence_score = prediction[0][user_index]

    # Kiểm tra nếu kết quả là Undefined
    if user_choice == undefined_class:
        print("Lựa chọn của người chơi là Undefined. Bỏ qua vòng này.")
        continue  # Bỏ qua vòng này và yêu cầu người chơi thử lại

    # AI chọn ngẫu nhiên một hành động
    ai_choice = random.choice(ai_choices)

    # Kiểm tra nếu AI chọn Undefined
    if ai_choice == undefined_class:
        print("Lựa chọn của AI là Undefined. Bỏ qua vòng này.")
        continue  # Bỏ qua vòng này và yêu cầu AI thử lại

    # Xác định kết quả của trò chơi
    result = determine_winner(user_choice, ai_choice)
    update_scores(result)

    # Phát âm thanh hiệu ứng tương ứng dựa trên kết quả
    if result == "Draw":
        draw_sound.play()
    elif result == "User Wins":
        win_sound.play()
    else:
        lose_sound.play()

    # Hiển thị kết quả trên terminal
    print(f"Người chơi: {user_choice}, AI: {ai_choice}, Kết quả: {result}")

    # Lấy kích thước của khung hình để tính toán vị trí căn giữa
    frame_height, frame_width, _ = frame.shape

    # Căn giữa các văn bản
    round_text = f"Round {current_round}/{total_rounds}"
    result_text = f"Result: {result}"
    ai_text = f"AI: {ai_choice}"
    user_text = f"User: {user_choice}"
    ai_score_text = f"AI Score: {ai_score}"
    user_score_text = f"User Score: {user_score}"

    # Vị trí cho các dòng văn bản
    round_pos = (frame_width // 2 - cv2.getTextSize(round_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2, 30)
    ai_pos = (10, 70)  # AI ở bên trái, gần sát trên cùng
    user_pos = (frame_width - cv2.getTextSize(user_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] - 10,
                70)  # User ở bên phải, gần sát trên cùng
    result_pos = (frame_width // 2 - cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] // 2,
                  frame_height - 60)  # Kết quả ở dưới cùng giữa trên User và AI score
    ai_score_pos = (frame_width - cv2.getTextSize(ai_score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] - 10,
                    frame_height - 30)  # AI score ở góc dưới bên phải
    user_score_pos = (10, frame_height - 30)  # User score ở góc dưới bên trái

    # Hiển thị các văn bản trên màn hình
    cv2.putText(frame, round_text, round_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, ai_text, ai_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, user_text, user_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, result_text, result_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, ai_score_text, ai_score_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, user_score_text, user_score_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Hiển thị khung hình camera
    cv2.imshow("Trò chơi Kéo-Bao-Búa", frame)

    # Hiển thị kết quả trong 3 giây trước khi tiếp tục
    cv2.waitKey(3000)

    # Yêu cầu người chơi tiếp tục vòng kế tiếp
    print("Nhấn Enter để chơi vòng tiếp theo, hoặc ESC để thoát.")
    key = cv2.waitKey(0)  # Đợi người chơi nhấn phím
    if key == 27:  # Nhấn ESC để thoát
        print("Người chơi thoát trò chơi.")
        break

    current_round += 1

# Dừng phát nhạc sau khi trò chơi kết thúc
pygame.mixer.music.stop()

# Hiển thị kết quả cuối cùng trên màn hình camera
frame_height, frame_width, _ = frame.shape

# Tạo các dòng văn bản để hiển thị kết quả cuối cùng
final_result_text = ""
if user_score > ai_score:
    final_result_text = "Congragulation! You're champion"
elif ai_score > user_score:
    final_result_text = "AI wins! You're stupid person!"
else:
    final_result_text = "DRAW!"

# Vị trí cho kết quả cuối cùng
final_result_pos = (frame_width // 2 - cv2.getTextSize(final_result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] // 2,
                    frame_height // 2)

# Hiển thị kết quả cuối cùng
cv2.putText(frame, final_result_text, final_result_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Hiển thị lại khung hình với kết quả cuối cùng
cv2.imshow("Trò chơi Kéo-Bao-Búa - Kết quả cuối cùng", frame)

# Đợi 10 giây để người chơi có thể nhìn thấy kết quả
cv2.waitKey(10000)

# Giải phóng tài nguyên
camera.release()
cv2.destroyAllWindows()
