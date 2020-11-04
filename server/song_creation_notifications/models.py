from django.db import models

from django.db.models.signals import pre_delete
from django.dispatch import receiver

from song_creation.signals import new_task
from song_creation.models import Song, Task


@receiver(new_task)
def handle_new_song(sender, **kwargs):
    song = kwargs['song']
    #task = kwargs['task']

    message = """Received a new request for a song with name {}.
    """.format(song.name)

    print(message)

@receiver(pre_delete, sender=Song)
def handle_deleted_songs(**kwargs):
    song = kwargs['instance']

    message = """song with name {} has been removed.
    """.format(song.name)

    print(message) 