from django.db import models

class Song(models.Model):
    name = models.CharField(max_length=255, blank=False)
    genre = models.CharField(max_length=255, blank=False)
    vocals = models.BooleanField(default=False)
    lyrics = models.BooleanField(default=False)
    soundcloud_track_id = models.CharField(max_length=255, blank=True)
    soundcloud_track_url = models.CharField(max_length=255, blank=True)
    date_created = models.DateTimeField(auto_now_add=True)
    date_modified = models.DateTimeField(auto_now=True)

class Task(models.Model):
    song = models.ForeignKey(Song, related_name="songs", on_delete=models.CASCADE)
    status = models.BooleanField(default=False)
    output = models.BinaryField(blank=True)
    date_created = models.DateTimeField(auto_now_add=True)
    date_modified = models.DateTimeField(auto_now=True)