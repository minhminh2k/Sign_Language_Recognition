import csv

# Đường dẫn tới file CSV của bạn
file_path = 'Choose.csv'

def getValue(col):
    # Đường dẫn tới file CSV của bạn
    file_path = 'Choose.csv'
    # Đọc dữ liệu từ file CSV
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if col == "Choose_label":
                choose_label_value = row["Choose_label"]
            elif col == "Click":
                choose_label_value = row["Click"]
            elif col == "Search":
                choose_label_value = row["Search"]
            elif col == "All":
                choose_label_value = row["All"]
            return choose_label_value

def changeValue( col, s):
    # Đường dẫn tới file CSV của bạn
    file_path = 'Choose.csv'
    # Đọc dữ liệu từ file CSV và lưu vào một danh sách
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Thay đổi giá trị của hàng đầu tiên trong cột col
    if col == "Choose_label":
        data[0]["Choose_label"] = s    
    elif col == "Click":
        data[0]["Click"] = s    
    elif col == "Search":
        data[0]["Search"] = s  
    elif col == "All":
        data[0]["All"] = s    

    # Ghi dữ liệu mới vào file CSV
    with open(file_path, mode='w', newline='') as file:
        fieldnames = ["Choose_label", "Click", "Search", "All"]  # Định nghĩa tên cột
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()  # Viết header

        # Viết dữ liệu
        for row in data:
            writer.writerow(row)

    #print("Dữ liệu đã được thay đổi và ghi vào file CSV.")

def createCsv():
    file_path = "Choose.csv"
    # Dữ liệu bạn muốn ghi vào file CSV
    data = [
        ["Choose_label", "Click", "Search", "All"],
        ["0", "0", "0", "1"]
    ]

    # Ghi dữ liệu vào file CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("File CSV đã được tạo thành công.")