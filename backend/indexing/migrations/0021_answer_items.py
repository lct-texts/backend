# Generated by Django 4.2.5 on 2023-10-01 11:15

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("indexing", "0020_answer_created"),
    ]

    operations = [
        migrations.AddField(
            model_name="answer",
            name="items",
            field=models.ManyToManyField(to="indexing.item"),
        ),
    ]