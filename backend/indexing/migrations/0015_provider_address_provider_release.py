# Generated by Django 4.2.5 on 2023-09-30 19:03

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("indexing", "0014_filterrange_values_amount"),
    ]

    operations = [
        migrations.AddField(
            model_name="provider",
            name="address",
            field=models.CharField(max_length=1000, null=True),
        ),
        migrations.AddField(
            model_name="provider",
            name="release",
            field=models.CharField(blank=True, max_length=1000, null=True),
        ),
    ]
