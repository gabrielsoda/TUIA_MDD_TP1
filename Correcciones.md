REHACER (Plazo Máximo 24/10)
* No tiene titulo de trabajo ni nombres de integrantes. Falta carátula.
* Hay una mezcla de comentarios en el código entre inglés y español ¿Por qué? además, hablan de manera personal? "# Adjust this to the maximum number of dimensions you want to test"
* No se debe hacer suposiciones infundamentadas al rellenar valores nulos, puede llevar a graves errores en la práctica: "Como no necesariamente el cultivo debe estar enfermo, vamos a considerar que los valores faltantes de la variable corresponden a cultivos saludables."
* Llenan el valor nulo en estadoEnfermedadesCultivo con una categoría nueva, lo cual no es correcto. Llenan el valor de tipoRiego con la categoría "desconocido". Introducir una categoría nueva no tiene un significado real en el fenómeno observado. Distorsiona los datos.
* No aclaran qué valor óptimo es por el método del codo, ni por varianza acumulada como tampoco por el método que eligieron incorporar: el Kaiser.
* En Isomap, la conclusión "Si bien ninguna combinación de hiperparámetro coincide con una separación de tipo de cultivo, podemos visualizar agrupaciones más marcadas con 6 dimensiones." ¿Cómo sería con 6 dimensiones?. Además, en los gráficos anteriores no se visualizan las "6 dimensiones". Al principio, hacen "dimensiones" con el gráfico del codo, no se entiende para qué lo aplican.
* Cuando hacen el código para probar los vecinos, dejan fijadas las dimensiones en 8.
* En t-SNE: Se deben mostrar las iteraciones probadas: "Luego de intentar con distintos valores de perplexidad, encontramos que en 20 se separan más los datos." No solo se debe variar perplejidad, sino también iteraciones, una a la vez.
* En K-Means se aplica PCA y no estaba solicitado.