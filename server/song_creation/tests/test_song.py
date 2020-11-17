import json
from rest_framework import status
from django.test import TestCase, Client
from django.urls import reverse
from ..models import Song
from ..serializers import SongSerializer

client = Client()

class SongTestCase(TestCase):

    def setUp(self):
        self.first_song = Song.objects.create(
            name='First test song')
        self.second_song = Song.objects.create(
            name='Second test song')
        self.song_payload_valid = {
            "data": {
                "type": "Song",
                "attributes": {
                    "name": "Test song",
                    "genre": "Rock",
                    "vocals": False,
                    "lyrics": True,
                    "soundcloud_track_id": "",
                    "soundcloud_track_url": ""
                }
            }
        }
        self.song_payload_invalid = {
            "data": {
                "type": "Song",
                "attributes": {
                    "name": "",
                }
            }
        }

    def test_get_all_songs(self):
        response = client.get(reverse('song-list'))
        songs = Song.objects.all()
        serializer = SongSerializer(songs, many=True)
        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_get_single_song(self):
        response = client.get(reverse('song-detail', kwargs={'pk': self.first_song.pk}))
        song = Song.objects.get(pk = self.first_song.pk)
        serializer = SongSerializer(song)
        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_get_invalid_single_song(self):
        response = client.get(reverse('song-detail', kwargs={'pk': 100}))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_create_single_song(self):
        response = client.post(
            reverse('song-list'),
            data=json.dumps(self.song_payload_valid),
            content_type='application/vnd.api+json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_create_invalid_song(self):
        response = client.post(
            reverse('song-list'),
            data=json.dumps(self.song_payload_invalid),
            content_type='application/vnd.api+json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)