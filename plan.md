
Notas importantes:
    - Pedir dataset world-view 3


Objetivos:

    - Conseguir, a partir de un conjnto de imagenes finito, poder sintetizar vistas de momentos que la camara no ha visto desde angulos que la camara no ha visto.

Pasos:
    - Conseguir imagenes geotagged de una región desde multiples satelites (diferentes resoluciones, diferentes momentos, diferentes angulos).
    - Preprocesar las imagenes para que queden patches iguales de la misma area. Hay que convertir de coordenadas pixel a coordenadas lat,lon.
    - Saber a qué coordenadas corresponde cada pixel de cada imagen.
    - Hacer un sistema de volume rendering para poder generar las imagenes. Es super importante que lo entiendas todo, porque es la base.
    - Entrenar el modelo con batches de pixeles. Testear con batches de pixeles diferentes a los de entrenamiento.
    
Descripción:
    - El modelo recibirá:
        - Posición (x, y, z):
        - Angulo (theta, phi) o un vector
        - Tiempo (dia del año, hora del dia, hora de salida del sol, hora de puesta del sol)
        - Posición del sol (azimuth, elevation)

    - El modelo dará:
        - Densidad
        - Color RGB

