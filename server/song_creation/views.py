from django.shortcuts import render

from .models import Song
from .models import Task
from .signals import new_task
from .serializers import SongSerializer, TaskSerializer
from rest_framework import viewsets

def get_songs(request):
    songs = Song.objects.all()
    return render(request, 'songs.html', {'songs': songs})

def get_song(request, id):
    song = Song.objects.get(pk=id)
    return render(request, 'song.html', {'song': song})

def create(request):
    song = Song(name=request.POST['name'], genre=request.POST['genre'], vocals=request.POST['vocals'], lyrics=request.POST['lyrics'])
    song.save()

    task = Task(song=song, status=False)
    task.save()

    new_task.send(sender=task, song=song, task=task)

    payload = {
      'song': song,
      'name': request.POST['name'],
      'genre': request.POST['genre'],
      'vocals': request.POST['vocals'],
      'lyrics': request.POST['lyrics']
    }
    return render(request, 'created.html', {'payload': payload})

class SongViewSet(viewsets.ModelViewSet):
    queryset = Song.objects.all()
    serializer_class = SongSerializer

class TaskViewSet(viewsets.ModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer
