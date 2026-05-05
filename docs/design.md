
# 内部設計

## モジュール図

* モジュール
    * pnq/__queries__.py: queries.py 用のテンプレート。型アノテーションなど複雑な一貫性のためにテンプレート化している。
    * pnq/queries.py: __queries__ から生成された実際のコード。型アノテーションや内部オブジェクトとの連携を行う結合層。pnq/_itertools/queryables を参照する 
* クラス・ファンクション
    * pnq/queries.Query: 基本の型
    * pnq/queries.PairQuery: キーバリュー向けの型
    * pnq/pnq/protocols.WrappedQuery: AST 解析のような機能があるクラス
        * pnq/_itertools/core.Query: WrappedQuery(PQuery)を継承。イテレータに関する基本的な実装が含まれる。
            * pnq/_itertools/queryables.Query: core.Query を継承し、さらに基本的な実装が含まれる。このクラスを規定に Map など具体的なクエリメソッドを実装する。ただし、このクラスには、_ait（非同期用イテレータクエリ）と _sit（同期用イテレータクエリ）の属性が用意され、そこに参照を埋める形。
    * pnq/_itertools
        * _async: 非同期用イテレータクエリ。queryables.Query から参照される
        * _sync_generate: 同期用イテレータクエリ。_async から自動生成される
        * _sync:  同期用イテレータクエリ。自動生成できない部分を管理する。queryables.Query から参照される
    * pnq/_itertools/generate_unasync.py: pnq/_itertools/_async から非同期コード（_sync_generate）を生成する。

## なぜ複雑か？

* 同期と非同期を一緒に管理しようとしている
* 同期と非同期用の型アノテーションを一括で管理しようとしている
* 非同期コードから同期コードを自動生成している
* その他、整理が進んでいない

## リファクタリング計画

* Python のバージョンアップで無理やり解決していた型アノテーションをスマートな形に寄せる
* queries.Query, core.Query, queryables.Query で Query クラスを別の名前で区別する。名前が被っているせいで混乱を招く
* Query は継承していたりするが、ネストを少なくする
* とにかく基底クラスを分かりやすくする
* sleep などクエリと領域が違うものを除去する

