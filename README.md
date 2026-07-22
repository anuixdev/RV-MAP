# RV-MAP (Rescue Vision Mapping)

**Sistema de Mapeo por Visión de Rescate y Análisis de Transitabilidad**

RV-MAP es una arquitectura algorítmica de visión computacional diseñada para el soporte logístico en entornos colapsados. El núcleo de *software*, la entrada de datos ópticos capturada por vehículos aéreos no tripulados (UAV) y, mediante inferencia tensorial basada en YOLOv8, segmenta las estructuras destruidas, vehículos y escombros. Posteriormente, aplica operaciones de dilatación morfológica para generar mapas de costes bidimensionales que delimitan las áreas de tránsito seguro frente a zonas de exclusión.

## Características Técnicas

* **Inferencia Neuronal:** Extracción de características topológicas en tiempo real utilizando la arquitectura YOLOv8.
* **Geometría de Penalización:** Implementación de gradientes de riesgo cónicos euclidianos para calcular márgenes de seguridad físicos en torno a obstáculos dinámicos.
* **Auditoría Visual:** Proyección de contornos geométricos sobre la matriz de transitabilidad para validación táctica humana.
* **Despliegue Interactivo:** Interfaz de línea de comandos (CLI) parametrizable y optimizada para operar en plataformas de *Edge Computing*.

## Documentación Técnica y Flujo de Procesamiento

La arquitectura opera mediante un conducto (*pipeline*) secuencial que transforma la captura óptica bruta en un plano de decisión matricial.

* **Entrada de Datos (Input):** El motor admite ortofotos o capturas cenitales en formatos de imagen estándar (`PNG`, `JPG`, `JPEG`). Para mantener la integridad topológica, se exige una resolución que permita distinguir características métricas del terreno.
* **Procesamiento Tensorial:** La red neuronal convolucional escruta el fotograma para extraer los contornos poligonales y generar máscaras binarias sobre los píxeles clasificados como entidades de alto riesgo.
* **Operaciones Morfológicas:** Sobre la segmentación resultante, se aplica un núcleo de convolución (*kernel*) cuyo diámetro de acción se define paramétricamente. Este paso dilata las dimensiones geométricas de los obstáculos, inyectando un margen de seguridad equivalente a la anchura del vehículo de rescate.
* **Exportación Matricial (Output):** El sistema sintetiza un mapa de transitabilidad y lo almacena en el disco local de forma asíncrona. Si la bandera de auditoría está activa, superpone adicionalmente el cálculo sobre el entorno original.
* **Aceleración por Hardware:** El núcleo algorítmico detecta de forma automática la disponibilidad de unidades de procesamiento tensorial (núcleos CUDA) en el sistema anfitrión. En caso de carencia, el motor conmuta internamente a la ejecución secuencial en hilos de CPU sin interrumpir el flujo logístico.

## Despliegue del Entorno Computacional

Para instanciar el entorno de ejecución, se recomienda aislar las dependencias mediante un entorno virtual (`venv` o `conda`). Ejecute los siguientes comandos en la terminal de su sistema:

```bash
# 1. Clonar el repositorio (via HTTP)
git clone https://github.com/anuixdev/RV-MAP.git
cd RV-MAP

# 1'. Clonar el repositorio (via SSH)
git@github.com:anuixdev/RV-MAP.git
cd RV-MAP

# 2. Crear el entorno virtual (denominado 'venv')
python -m venv venv

# 3. Activar el entorno virtual
# -> En sistemas Windows (Command Prompt o PowerShell):
.\venv\Scripts\activate
# -> En sistemas basados en Unix (Linux / macOS):
source venv/bin/activate

# 4. Instalar dependencias matriciales y de aprendizaje profundo
pip install -r requirements.txt

# 5. Comando para ejecutar y ver resultados:
python main_rvmap.py --args (-i, -o, -m, -r, --show)
