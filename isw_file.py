from typing import List


def read_from_isw_file(filename: str) -> List[float]:
    """Чтение выборки из файла формата ISW.

    :param filename: имя файла
    :return: выборка
    """
    xs = []
    with open(filename, 'r') as file:
        file.readline()
        file.readline()
        while True:
            string_x = file.readline()
            if not string_x:
                break
            xs.append(float(string_x))
    return xs


def save_to_isw_file(data: List[float], filename: str, description: str):
    """Записать выборки в файл формата ISW.

    :param data: набор данных
    :param filename: имя файла
    :param description: описание записываемого набора данных
    :return:
    """
    with open(filename, 'w') as file:
        file.write(description + '\n')
        file.write(f"0 {len(data)}\n")
        for val in data:
            file.write(f"{val}\n")
