# Generated by Django 4.2.5 on 2023-09-30 11:34

import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("indexing", "0011_alter_provider_region"),
    ]

    operations = [
        migrations.CreateModel(
            name="FilterRange",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("okved", models.CharField()),
                ("min_value", models.IntegerField(blank=True, null=True)),
                ("max_value", models.IntegerField(blank=True, null=True)),
                (
                    "categories",
                    django.contrib.postgres.fields.ArrayField(
                        base_field=models.CharField(), blank=True, null=True, size=None
                    ),
                ),
                ("filter", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="indexing.filter")),
            ],
        ),
    ]