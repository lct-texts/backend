# Generated by Django 4.2.5 on 2023-09-28 17:24

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("indexing", "0002_alter_item_link_to_source_alter_item_okpd2_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="item",
            name="okpd2",
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]
