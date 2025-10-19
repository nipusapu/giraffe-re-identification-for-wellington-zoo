from rest_framework import serializers
from .models import ReIDResult, Animal

class AnimalSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Animal
        fields = ['id', 'code', 'description', 'created_at', 'updated_at']

class ReIDResultSerializer(serializers.ModelSerializer):
    predicted_animal = AnimalSerializer(read_only=True)

    class Meta:
        model  = ReIDResult
        fields = [
            'id', 'image', 'status',
            'predicted_animal', 'votes_json',
            'created_at', 'updated_at',
        ]
