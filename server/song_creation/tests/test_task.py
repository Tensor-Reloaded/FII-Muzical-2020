import json
from rest_framework import status
from rest_framework.request import Request
from django.test import TestCase, Client
from rest_framework.test import APIRequestFactory
from django.urls import reverse
from ..models import Task, Song
from ..serializers import TaskSerializer

client = Client()

factory = APIRequestFactory()
request = factory.get('/')

serializer_context = {
    'request': Request(request),
}

class TaskTestCase(TestCase):

    def setUp(self):
        test_song = Song.objects.create(name='Test')
        self.first_task = Task.objects.create(
            song=test_song)
        self.second_task = Task.objects.create(
            song=test_song)
        self.task_payload_valid = {
            "data": {
                "type": "Task",
                "attributes": {
                    "song": "http://127.0.0.1:8000/songs/1/"
                }
            }
        }
        self.task_payload_invalid = {
            "data": {
                "type": "Task",
                "attributes": {
                    "invalid_field": False,
                }
            }
        }

    def test_get_all_tasks(self):
        response = client.get(reverse('task-list'))
        tasks = Task.objects.all()
        serializer = TaskSerializer(tasks, many=True, context=serializer_context)
        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_get_single_task(self):
        response = client.get(reverse('task-detail', kwargs={'pk': self.first_task.pk}))
        task = Task.objects.get(pk = self.first_task.pk)
        serializer = TaskSerializer(task, context=serializer_context)
        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_get_invalid_single_task(self):
        response = client.get(reverse('task-detail', kwargs={'pk': 100}))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_create_single_task(self):
        response = client.post(
            reverse('task-list'),
            data=json.dumps(self.task_payload_valid),
            content_type='application/vnd.api+json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_create_invalid_task(self):
        response = client.post(
            reverse('task-list'),
            data=json.dumps(self.task_payload_invalid),
            content_type='application/vnd.api+json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)