# Generated by Django 4.2.5 on 2023-09-30 08:30

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("indexing", "0009_provider_okved_provider_rusprofile_link_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="provider",
            name="kapital",
            field=models.CharField(blank=True, null=True),
        ),
    ]
