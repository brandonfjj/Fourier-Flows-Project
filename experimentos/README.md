# Experimentos: Fourier Flows para datos sintéticos de PMU

Scripts de la Investigación 2 (Proyecto Eléctrico, UCR). Aplican Fourier Flows (Alaa, ICLR 2021) a señales de
µPMU simuladas, para compararlo con un TimeGAN con restricciones físicas. El modelo base es el del repositorio
original (`SequentialFlows.py`); aquí solo se usa la subclase `FourierFlowSafe` de `ff_safe.py`, que evita una
división por cero al normalizar el espectro.

**Los datos (`data/PMU_AlexyDonald2.csv`) son confidenciales y no están en el repositorio.** Todo lo que se genera
a partir de ellos va a `results/`, que está ignorado por git.

## Requisitos

- Entorno conda con Python 3.6, PyTorch 1.3.1 y numpy/pandas/matplotlib de esa época (`py36_ff`).
- El CSV de datos en `data/PMU_AlexyDonald2.csv`.
- Los scripts calculan la raíz del repositorio a partir de su propia ubicación, así que se pueden ejecutar
  desde cualquier carpeta.

## Estructura

```
experimentos/
├── pmu_pipeline.py          entrena un flujo por canal (20 canales) y guarda el sintético desnormalizado
├── validar_salida.py        integridad, rango, estadísticos, rugosidad y distancia espectral por canal
├── graficas_comparacion.py  figuras real vs sintético (ventanas, PCA, espectro, histogramas, métricas)
├── listar_corridas.py       tabla resumen de todas las corridas (results/corridas.csv)
├── correr_corrida.sh        encadena los cuatro pasos anteriores en una carpeta por corrida
├── analisis_rep.py          análisis de reproducibilidad y de las métricas de una corrida repetida
├── (otros .py)              estudios previos: ventanas de un ciclo, alineación de fase, tres fases,
│                            semillas, ventanas de 48 y diagnósticos de los datos
└── exploratorios/           primeras pruebas y versiones antiguas

results/                     (no versionado)
├── runs/<id>/               una carpeta por entrenamiento completo
├── pruebas/                 pruebas cortas del pipeline
├── exploratorios/           salidas de los estudios previos, una carpeta por estudio (ver su LEEME.txt)
└── corridas.csv             tabla generada por listar_corridas.py
```

El identificador de una corrida es `s<semilla>_e<épocas>_lr<lr>_p<paso>_<etiqueta>`, por ejemplo
`s0_e2000_lr1e-4_p5_orig`. Cada carpeta de corrida contiene el sintético (`pmu_sintetico.npz` y su `.json` con la
configuración), los registros (`pipeline.log`, `corrida.log`, `comando.txt`), la validación (`validacion.csv`) y
las figuras (`figs/`).

## Lanzar una corrida

```bash
# Ensayo: muestra la carpeta y los comandos sin ejecutar nada
experimentos/correr_corrida.sh --etiqueta s1 --semilla 1 --dry-run

# Corrida real en segundo plano (unos 23 min con 6 procesos)
nohup experimentos/correr_corrida.sh --etiqueta s1 --semilla 1 > /dev/null 2>&1 &

# Seguir el avance (Ctrl+C sale del visor, no detiene la corrida)
tail -F results/runs/s1_e2000_lr1e-4_p5_s1/corrida.log
```

Opciones: `--semilla`, `--epocas`, `--lr`, `--paso`, `--workers`, `--n_muestras`, `--etiqueta` (obligatoria) y
`--dry-run`. El script se detiene en el primer paso que falle y no sobrescribe una carpeta que ya tenga contenido.

Con la misma semilla y configuración el resultado es idéntico bit a bit; para medir la variabilidad entre
corridas hay que cambiar la semilla.

## Listar las corridas

```bash
python experimentos/listar_corridas.py
```

Imprime una línea por corrida (fecha, parámetros, forma, tiempo, reintentos, pérdida final media y si tiene
validación y figuras) y la guarda en `results/corridas.csv`.
