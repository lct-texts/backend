# Generated by Django 4.2.5 on 2023-09-28 17:24

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("indexing", "0003_alter_item_okpd2"),
    ]

    operations = [
        migrations.AlterField(
            model_name="item",
            name="name",
            field=models.CharField(max_length=1000),
        ),
    ]
