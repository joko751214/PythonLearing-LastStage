# -*- coding: utf-8 -*-
# Generated by Django 1.9 on 2018-04-03 08:42
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('stockapp', '0009_auto_20180329_1512'),
    ]

    operations = [
        migrations.CreateModel(
            name='DealStock',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('dealno', models.IntegerField(verbose_name=b'\xe6\x88\x90\xe4\xba\xa4\xe5\x8d\x95\xe5\x8f\xb7')),
                ('number', models.IntegerField(verbose_name=b'\xe8\x82\xa1\xe7\xa5\xa8\xe7\xbc\x96\xe7\xa0\x81')),
                ('damount', models.IntegerField(default=100, verbose_name=b'\xe6\x88\x90\xe4\xba\xa4\xe6\x95\xb0\xe9\x87\x8f')),
                ('totles', models.FloatField(verbose_name=b'\xe6\x88\x90\xe4\xba\xa4\xe9\x87\x91\xe9\xa2\x9d')),
                ('time', models.DateTimeField(auto_now_add=True, verbose_name=b'\xe6\x88\x90\xe4\xba\xa4\xe6\x97\xb6\xe9\x97\xb4')),
                ('buser', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='buser', to=settings.AUTH_USER_MODEL)),
                ('suser', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='suser', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['id'],
                'verbose_name': '\u6210\u4ea4\u5355',
                'verbose_name_plural': '\u6210\u4ea4\u5355',
            },
        ),
        migrations.AddField(
            model_name='hold',
            name='frozen_amount',
            field=models.IntegerField(default=100, verbose_name=b'\xe5\x86\xbb\xe7\xbb\x93\xe6\x95\xb0\xe9\x87\x8f'),
        ),
        migrations.AlterField(
            model_name='bosstock',
            name='state',
            field=models.IntegerField(choices=[(b'0', b'deity'), (b'1', b'deal'), (b'2', b'delete'), (b'3', b'cancel')], default=0, verbose_name=b'\xe7\x8a\xb6\xe6\x80\x81'),
        ),
        migrations.AlterField(
            model_name='bosstock',
            name='time',
            field=models.DateField(auto_now_add=True, verbose_name=b'\xe6\x8c\x82\xe5\x8d\x95\xe6\x97\xb6\xe9\x97\xb4'),
        ),
    ]