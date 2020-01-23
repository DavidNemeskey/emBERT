#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stuff related to model downloading."""

import os
import re
from typing import List
from urllib.request import urlopen, urlretrieve

from github import Github, UnknownObjectException
from progressbar import ProgressBar


class ReporthookProgressBar:
    """
    Original Source:
    https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve/53643011#53643011
    """
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_apache_dir(dir_url: str, output_directory: str):
    """
    Downloads a directory that has the usual Apache directory listing. Not
    recursive.

    :param dir_url: the URL of the directory.
    :param output_directory: the name of the output directory. If it doesn't
                             exist, it will be created.
    """
    trp = re.compile('<tr>(.+?)</tr>', re.I)
    hrefp = re.compile(r'<a href="([^"]+)"\s*>\1<', re.I)
    dirp = re.compile(r'alt="\[(?:PARENT)?DIR\]"', re.I)

    html = urlopen(dir_url).read().decode('utf-8')
    files = []
    for tr in map(lambda trm: trm.group(1), trp.finditer(html)):
        if not dirp.search(tr):
            m = hrefp.search(tr)
            if m:
                files.append(m.group(1))

    download_files(files, dir_url, output_directory)


def download_github_dir(repository: str, git_directory: str,
                        output_directory: str, branch: str = 'master'):
    """
    Downloads a directory from GitHub.

    :param repository: the _full_ name of the GitHub repository; i.e. in the
                       {user}/{repo} format. It must be public.
    :param git_directory: the name of the directory to download.
    :param output_directory: the name of the output directory. If it doesn't
                             exist, it will be created.
    :param branch: the Git branch to download from.
    """
    g = Github()
    repos = g.search_repositories(repository).get_page(0)
    if not repos:
        raise ValueError(f'No such repository found: {repository}')
    if len(repos) > 1:
        raise ValueError(f'More than one repositories match to {repository}')

    dir_url = f'https://github.com/{repository}/raw/{branch}/{git_directory}'
    try:
        git_files = [f.name for f in repos[0].get_contents(git_directory)]
        download_files(git_files, dir_url, output_directory)
    except UnknownObjectException:
        raise ValueError(f'Directory {git_directory} not found in {repository}')


def download_files(files: List[str], dir_url: str, output_directory: str):
    """
    Downloads a list of files from a server.

    :param files: the list of file names to download.
    :param dir_url: the url of the directory that stores the files.
    :param output_directory: the name of the output directory. If it doesn't
                             exist, it will be created.
    """
    if files:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        for file in files:
            url = f'{dir_url}/{file}'
            print('Downloading ', url)
            urlretrieve(url, os.path.join(output_directory, file),
                        ReporthookProgressBar())
            print()
