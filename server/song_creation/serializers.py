from rest_framework_json_api import serializers
from .models import Song, Task

class SongSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Song
        fields = ('name', 'genre', 'vocals', 'lyrics', 'soundcloud_track_id', 'soundcloud_track_url', 'date_created', 'date_modified')

class TaskSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Task
        fields = ('status', 'output', 'date_created', 'date_modified', 'song')