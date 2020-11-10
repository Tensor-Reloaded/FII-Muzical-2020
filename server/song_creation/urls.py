from django.urls import path, include
from rest_framework import routers
from .views import get_songs, get_song, create, SongViewSet, TaskViewSet

router = routers.DefaultRouter()
router.register(r'songs', SongViewSet)
router.register(r'tasks', TaskViewSet)

urlpatterns = [
    # path('songs/', get_songs, name="songs_view"),
    # path('songs/<int:id>', get_song, name="song_view"),
    # path('songs/create', create, name="create_view"),
    path('', include(router.urls)),
]