import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.fftpack import fft
import mne
from moabb.datasets import BNCI2014_002
from moabb.paradigms import MotorImagery

# ==========================
# CRIAR PASTA PARA SALVAR OS RESULTADOS
# ==========================
output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================
# CARREGAR O DATASET
# ==========================
print("Carregando dataset BNCI2014-002...")
dataset = BNCI2014_002()
paradigm = MotorImagery()
X, labels, metadata = paradigm.get_data(dataset, return_epochs=True)

# ==========================
# INFORMAÇÕES BÁSICAS
# ==========================
data = X.get_data()  # Converte para um array NumPy
sfreq = X.info['sfreq']  # Obtém a frequência de amostragem corretamente
duracao_real = data.shape[2] / sfreq  # Duração dos dados carregados

print(f"Número de amostras (trials): {data.shape[0]}")
print(f"Número de canais (eletrodos): {data.shape[1]}")
print(f"Duração real dos dados: {duracao_real:.3f} segundos")
print(f"Frequência de amostragem: {sfreq} Hz")
print("Classes disponíveis:", np.unique(labels))

# ==========================
# DOMÍNIO DO TEMPO - SINAL BRUTO
# ==========================
tempo = np.arange(data.shape[2]) / sfreq  # Criando vetor de tempo

print("Gerando gráfico do sinal no domínio do tempo...")
plt.figure(figsize=(12, 5))
plt.plot(tempo, data[0, 0, :], label="Canal 1 (EEG)", color='red')

# Adicionando marcações visuais dos eventos do experimento
plt.axvline(x=0.05, color='black', linestyle='--',
            label='Início do Trial (Cruz)')
plt.axvline(x=2, color='blue', linestyle='--', label='Beep')
plt.axvline(x=3, color='green', linestyle='--', label='Cue Visual (Início)')
plt.axvline(x=duracao_real, color='purple', linestyle='--',
            label='Fim da Imaginação Motora (Dados disponíveis)')

plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (u𝑉)')
plt.title('Sinal EEG - Domínio do Tempo')
plt.legend()
# Mostrando o intervalo completo do protocolo, apesar da limitação dos dados
plt.xlim(0, 8)
plt.savefig(os.path.join(output_dir, "dominio_tempo.png"))
plt.close()

# ==========================
# DOMÍNIO DA FREQUÊNCIA (FFT)
# ==========================
print("Gerando gráfico do espectro de frequências...")
eeg_fft = np.abs(fft(data[0, 0, :]))  # FFT do primeiro canal
frequencias = np.fft.fftfreq(len(eeg_fft), d=1/sfreq)

plt.figure(figsize=(12, 5))
plt.plot(frequencias[:200], eeg_fft[:200])  # Mostrar até 100 Hz
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.title('Transformada de Fourier (FFT) - Domínio da Frequência')
plt.savefig(os.path.join(output_dir, "dominio_frequencia.png"))
plt.close()

# ==========================
# DOMÍNIO TEMPO-FREQUÊNCIA (Espectrograma)
# ==========================
print("Gerando espectrograma...")
f, t, Sxx = scipy.signal.spectrogram(
    data[0, 0, :], fs=sfreq, nperseg=512, noverlap=256)

# Filtrar para exibir apenas até 50 Hz
mask = f <= 50  # Seleciona apenas frequências até 50 Hz

plt.figure(figsize=(12, 5))
plt.pcolormesh(t, f[mask], Sxx[mask, :], shading='gouraud')
plt.ylabel('Frequência (Hz)')
plt.xlabel('Tempo (s)')
plt.title('Espectrograma - Domínio Tempo-Frequência (até 50 Hz)')
plt.colorbar(label="Potência")
plt.savefig(os.path.join(output_dir, "dominio_tempo_frequencia.png"))
plt.close()

# ==========================
# DOMÍNIO ESPACIAL (Mapa Topográfico)
# ==========================
print("Gerando mapa topográfico...")

# Criar uma montagem padrão
montage = mne.channels.make_standard_montage("standard_1020")

# Obter nomes dos canais do dataset
eeg_channel_names = X.info["ch_names"]

# Criar um dicionário para renomear os canais
rename_dict = {eeg_channel_names[i]: montage.ch_names[i]
               for i in range(len(eeg_channel_names))}

# Renomear os canais no objeto Epochs
X.rename_channels(rename_dict)

# Aplicar montagem corretamente
X.set_montage(montage)

# Simular valores médios de atividade nos eletrodos
data_topo = np.random.rand(len(eeg_channel_names)) * 10

# Criar figura para topografia
fig, ax = plt.subplots(figsize=(6, 6))
im, _ = mne.viz.plot_topomap(
    data_topo[:len(eeg_channel_names)], X.info, axes=ax, show=False)

# Adicionar barra de cores
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Potencial (µV)", fontsize=12)

# Adicionar título ao gráfico
plt.title("Atividade Cortical - Domínio Espacial", fontsize=14)

# Salvar imagem na pasta results
output_path = os.path.join(output_dir, "dominio_espacial.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Todos os gráficos foram salvos na pasta '{output_dir}'!")
