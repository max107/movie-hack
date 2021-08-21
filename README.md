<h4>Реализованная функциональность</h4>
<ul>
    <li>Подгрузка и выгрузка видео;</li>
    <li>Разбивка на кадры;</li>
    <li>Сегментирование обьектов на каждом кадре;</li>
</ul> 
<h4>Особенность проекта в следующем:</h4>
<ul>
 <li>Обученная модель для распознования номеров машин;</li>
 <li>Размитие отдельных обьектов с помощью функции Гаусса;</li>
 </ul>
<h4>Основной стек технологий:</h4>
<ul>
    <li>Git, Ubuntu, Aws s3</li>
    <li>Python, Yolov5, opencv</li>
    <li>Golang, sqs, consumers </li>
 </ul>
<h4>Демо</h4>
<p>Демо сервиса доступно по адресу: https://t.me/hack0820_bot </p>
<p>Use case - загрузка видео, выбор параметров, в ответ Вам придет отредактированное видео.</p>
<p>Просьба грузить небольшие видео (<50mb),  сервис тестовый с ограниченными серверными ресурсами. </p>

СРЕДА ЗАПУСКА
------------
1) развертывание сервиса производится на debian-like linux (debian 9+);
2) требуется установленный web-сервер с поддержкой PHP(версия 7.4+) интерпретации (apache, nginx);
3) требуется установленная СУБД MariaDB (версия 10+);
4) требуется установленный пакет name1 для работы с...;


УСТАНОВКА
------------
### Ветка для локального тестирования

- https://github.com/panichmaxim/movie-hack/tree/colab
  
### Playbook в goole colab

- Collab folder: https://drive.google.com/drive/folders/1a0PQ0mGkESKa7HVjnITMSaD4OdttuGJY?usp=sharing


### Установка зависимостей проекта

~~~
pip install requirements.txt
~~~


### Команды

- ./car_number_detection/yolo/main.py --inputVideoPath video.mp4 --outputFilePath result.avi --modelPath model.weights


РАЗРАБОТЧИКИ

<h4>Фалалеев Максим fullstack https://t.me/a15646 </h4>
<h4>Панич Максим ML developer https://t.me/panichm </h4>
<h4>Иван Подмогильный ML developer https://t.me/gustoasa </h4>