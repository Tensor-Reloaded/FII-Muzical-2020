from django.urls import path
from .views import get_songs, get_song, create

urlpatterns = [
    path('songs/', get_songs, name="songs_view"),
    path('songs/<int:id>', get_song, name="song_view"),
    path('songs/create', create, name="create_view"),
]