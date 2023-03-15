# zerotsuku2_mokumoku
DSL主催の、「ゼロから作るDeep Learning 2」のもくもく会用のリポジトリになります。

# 概要
## 日程
3/14（火）〜3/30（木） 22:00〜23:00

## 扱う図書
ゼロから作るDeep Learning 2

## 参加方法
このGoogle フォームから申し込みをお願いします！途中参加も大歓迎です！
https://forms.gle/nJSx2xwpdBYoghT66

参加申し込み確認後、Slackチャンネルに招待します。

## 【参考資料】
- AIcia SolidさんのYouTube https://www.youtube.com/@AIcia_Solid
- GitHubアカウント作成方法 https://qiita.com/ayatokura/items/9eabb7ae20752e6dc79d
- GitHubのフォークからプッシュまで https://docs.github.com/ja/get-started/quickstart/fork-a-repo
- 今日のスライド

## 【GitHub設定手順】
1. GitHubでDSLの今回のリポジトリを検索（URL：https://github.com/DataScienceLeagueJapan/zerotsuku2_mokumoku）
2. 画面右上の「fork」からforkする
3. クローンする：`git clone https://github.com/[GitHubユーザー名]/zerotsuku2_mokumoku.git`
4. クローンしたフォルダ内に移動：`cd [クローンしたフォルダ名]`
5. upstream登録する：`git remote add upstream https://github.com/DataScienceLeagueJapan/zerotsuku2_mokumoku.git`
6. `git remote -v` で確認。以下のようになっていればOK。

origin https://github.com/[GitHubユーザー名]/zerotsuku2_mokumoku.git (fetch)  
origin https://github.com/[GitHubユーザー名]/zerotsuku2_mokumoku.git (push)  
upstream https://github.com/DataScienceLeagueJapan/zerotsuku2_mokumoku.git (fetch)  
upstream https://github.com/DataScienceLeagueJapan/zerotsuku2_mokumoku.git (push)  

7. フォルダを作成し、その中に何かファイルを作る
8. `git add .`
9. `git commit -m '[コミットメッセージ（日付 何ページまで進んだか 名前）]'`
10. `git push origin main`
11. 自分のGitHubページの今イベントのリポジトリに行き、「Pull requests」→「New pull request」→「Create pull request」→「Create pull request」
※pushするものは自分の名前をフォルダ名にして一つにまとめてpushしてください

【初回参加時にやること】
- GitHubの設定を行う
- 本を進める
- GitHubにpushしてpull requestを出す


