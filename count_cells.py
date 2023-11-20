import cv2
import numpy as np
from matplotlib import pyplot as plt

def count_cells_watershed_otimizado(caminho_da_imagem, contagem_verdadeira, params):
    # Ler a imagem
    imagem = cv2.imread(caminho_da_imagem)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Descompactar os parâmetros
    k_size, iteracoes, limiar = params

    # Aplicar o GaussianBlur para reduzir o ruído e melhorar a segmentação
    borrado = cv2.GaussianBlur(cinza, (5, 5), 0)

    # Aplicar threshold adaptativo
    _, thresh = cv2.threshold(borrado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Realizar operações morfológicas para limpar a imagem
    kernel = np.ones((k_size, k_size), np.uint8)
    abertura = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iteracoes)
    sure_bg = cv2.dilate(abertura, kernel, iterations=3)

    # Encontrar o primeiro plano (foreground) usando a transformada de distância
    dist_transform = cv2.distanceTransform(abertura, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, limiar * dist_transform.max(), 255, 0)

    # Subtrair o primeiro plano do plano de fundo para obter a região desconhecida
    sure_fg = np.uint8(sure_fg)
    desconhecida = cv2.subtract(sure_bg, sure_fg)

    # Rotular marcadores para o algoritmo de watershed
    _, marcadores = cv2.connectedComponents(sure_fg)
    marcadores += 1
    marcadores[desconhecida == 255] = 0

    # Aplicar o algoritmo de watershed
    cv2.watershed(imagem, marcadores)
    imagem[marcadores == -1] = [0, 0, 255]  # Marcar as bordas identificadas pelo watershed

    # Contar o número de células (excluindo o fundo)
    num_cells = len(np.unique(marcadores)) - 1

    # Usar a diferença absoluta entre a contagem predita e a contagem verdadeira como a perda
    perda = abs(contagem_verdadeira - num_cells)

    # # Exibir o resultado
    # plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    # plt.title(f"Contagem de Células com Watershed Otimizado ({num_cells})")
    # print('>>> Número de células: ', num_cells)
    plt.show()

    return perda
def count_cells_with_best_parameters(image_path, best_params):
    # Ler a imagem
    image = cv2.imread(image_path)
    cinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Descompactar os melhores parâmetros
    k_size, iteracoes, limiar = best_params

    # Aplicar o GaussianBlur para reduzir o ruído e melhorar a segmentação
    borrado = cv2.GaussianBlur(cinza, (5, 5), 0)

    # Aplicar threshold adaptativo
    _, thresh = cv2.threshold(borrado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Realizar operações morfológicas para limpar a imagem
    kernel = np.ones((k_size, k_size), np.uint8)
    abertura = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iteracoes)
    sure_bg = cv2.dilate(abertura, kernel, iterations=3)

    # Encontrar o primeiro plano (foreground) usando a transformada de distância
    dist_transform = cv2.distanceTransform(abertura, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, limiar * dist_transform.max(), 255, 0)

    # Subtrair o primeiro plano do plano de fundo para obter a região desconhecida
    sure_fg = np.uint8(sure_fg)
    desconhecida = cv2.subtract(sure_bg, sure_fg)

    # Rotular marcadores para o algoritmo de watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[desconhecida == 255] = 0

    # Aplicar o algoritmo de watershed
    cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]  # Marcar as bordas identificadas pelo watershed

    # Contar o número de células (excluindo o fundo)
    num_cells = len(np.unique(markers)) - 1

    # Exibir o resultado
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Contagem de Células com Melhores Parâmetros")
    plt.show()

    return num_cells

if __name__ == '__main__':

    caminho_da_imagem = 'test2.png'

    # Contagem verdadeira para a imagem de exemplo
    contagem_verdadeira = 42

    # Parâmetros a serem otimizados [tamanho_kernel, iterações, limiar]
    params_a_otimizar = [(3, 2, 0.7), (5, 2, 0.7), (3, 3, 0.8)]

    # Encontrar os parâmetros que minimizam a perda
    melhores_params = min(params_a_otimizar, key=lambda params: count_cells_watershed_otimizado(caminho_da_imagem, contagem_verdadeira, params))
    num_cells_best_params = count_cells_with_best_parameters(caminho_da_imagem, melhores_params)

    print(f">>> Melhores Parâmetros: {melhores_params}")
    print(f">>> Número total de células: {num_cells_best_params}")