#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stuff related to model downloading."""

import os
from urllib.request import urlretrieve

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


def download_github_dir(repository: str, git_directory: str,
                        output_directory: str, branch: str = 'master'):
    """
    Downloads a directory from GitHub.

    :param repository: the _full_ name of the GitHub repository; i.e. in the
                       {user}/{repo} format. It must be public.
    :param git_directory: the name of the directory to download.
    :param output_directory: the name of the output directory. If it doesn't
                             exist, it is created.
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
        git_files = repos[0].get_contents(git_directory)
    except UnknownObjectException:
        raise ValueError(f'Directory {git_directory} not found in {repository}')

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    for git_file in git_files:
        url = f'{dir_url}/{git_file.name}'
        print('Downloading ', url)
        urlretrieve(url, os.path.join(output_directory, git_file.name),
                    ReporthookProgressBar())
        print()
