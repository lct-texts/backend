# Generated by Django 4.2.5 on 2023-11-05 13:50

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("indexing", "0025_conversation_can_promote_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="conversation",
            name="sentiment_score",
            field=models.DecimalField(decimal_places=2, default=0, max_digits=3),
        ),
    ]