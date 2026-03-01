# Workshop 1: MiniTorch - Framework de Deep Learning desde Cero

En este primer workshop desarrollamos un framework minimalista de Deep Learning inspirado en PyTorch, pero construido desde cero.  

El objetivo no era solo entrenar una red neuronal, sino entender realmente qué está pasando internamente en cada paso: cómo fluye la información hacia adelante, cómo se calculan los gradientes y cómo se actualizan los parámetros sin depender de autograd.


## Descripción

En este taller implementamos manualmente los componentes esenciales de un framework de Deep Learning:

- Capas neuronales (Linear, ReLU, Dropout, BatchNorm)  
- Función de pérdida (Cross-Entropy)  
- Propagación hacia adelante (forward pass)  
- Retropropagación manual (backpropagation, sin autograd)  
- Optimización con SGD implementada desde cero  
- Loop completo de entrenamiento y validación

La idea fue reconstruir las bases de lo que normalmente hace una librería como PyTorch “por debajo”.

## Archivos del Proyecto

- `MiniTorchWorkshop.ipynb`: Notebook principal con la implementación, experimentos y análisis de resultados.  
- `minitorch.py`: Archivo que contiene el framework con todas las clases de capas y funciones implementadas.  
- `README.md`: Documentación del workshop (Este archivo).  

## Objetivos de Aprendizaje

1. Comprender la propagación hacia adelante (forward pass).
2. Implementar la retropropagación (backpropagation) manualmente.
3. Entender el cálculo de gradientes sin usar librerías de autograd.
4. Programar el algoritmo de optimización SGD.
5. Construir, entrenar y evaluar una red neuronal completa.

Más que usar herramientas listas, el enfoque fue entender cada derivada y cada actualización de parámetros.

## Dataset

Trabajamos con el dataset **MNIST**, un conjunto clásico de dígitos escritos a mano (28×28 píxeles, 10 clases).
- Training: 48,000 imágenes (80%)
- Validation: 12,000 imágenes (20%)
- Test: 10,000 imágenes

## Arquitecturas Implementadas

En el notebook se experimenta con distintas arquitecturas para comparar desempeño, entre ellas:

### Arquitectura Base (Regresión Softmax)
```
Input (784) → Linear (1024) → Linear (10) → Softmax
```

### Arquitecturas con Activaciones No Lineales y Regularización
```
Input (784) → Linear (512) → ReLU → Dropout → 
Linear (256) → ReLU → Linear (10) → Softmax
```

## Componentes Implementados

### Capas
- **Linear**: Capa completamente conectada con forward/backward manual.
- **ReLU**: Función de activación con derivada implementada.
- **Dropout**: Regularización con inverted dropout.
- **BatchNorm1D**: Normalización por lotes.

### Función de Pérdida
- **CrossEntropyFromLogits**: Combinación de Softmax + Cross-Entropy

### Contenedor
- **Net**: Clase secuencial para apilar capas con modos train/eval

## Resultados

Los mejores resultados obtenidos se documentan al final del notebook, incluyendo:
- Arquitectura seleccionada
- Accuracy en conjunto de validación y test
- Análisis de convergencia
- Comparación entre arquitecturas

## Uso de herramientas de IA  

Durante el desarrollo de los talleres hemos utilizado distintas herramientas de IA como apoyo. Estas NO reemplazan el trabajo propio, sino que nos han servido como soporte para entender mejor los conceptos, depurar errores y mejorar la documentación.

### Claude (Anthropic)  

Lo hemos usado principalmente como apoyo conceptual y para mejorar la redacción. Nos ha ayudado a:

- Entender conceptos teóricos de Deep Learning.  
- Organizar y estructurar mejor la documentación.  
- Aclarar dudas sobre implementaciones específicas.  

### GitHub Copilot  

Ha sido útil como asistente de código en tiempo real:

- Autocompletado inteligente.  
- Sugerencias de implementación.  
- Apoyo en correcciones sintácticas.  
- Optimización de fragmentos de código.  

### ChatGPT (OpenAI)  

Lo hemos utilizado sobre todo para resolver dudas técnicas puntuales:

- Errores en la terminal.  
- Comandos de Git y GitHub.  
- Configuración del entorno de desarrollo.  
- Debugging de problemas específicos.  

## Ejecución

```bash
# Instalar dependencias
pip install torch torchvision numpy matplotlib tqdm

# Ejecutar notebook
jupyter notebook MiniTorchWorkshop.ipynb
```

## Competencia en Hugging Face

El modelo final fue subido a [MLEAFIT](https://huggingface.co/MLEAFIT) para competir con otros estudiantes usando un conjunto de test con datos no vistos.

## Referencias

- [Documentación oficial de PyTorch](https://pytorch.org/docs/stable/index.html)  
- [Repositorio del curso (material del profesor, diapositivas y enunciados)](https://github.com/jdmartinev/ArtificialIntelligenceIM/tree/main)  
- [MLEAFIT en Hugging Face (plataforma para competencia de modelos)](https://huggingface.co/MLEAFIT)  

## Autores

- Juan Pablo Castaño  
- Sara Sofía Quintero  
- Karol Vanessa Cuello

Estudiantes de Ingeniería Matemática – Universidad EAFIT  