from django.dispatch import Signal

new_task = Signal(providing_args=["song", "task"])