from today_eat_what import run_workflow


def main():
    # Kick off the LangGraph workflow once.
    final_state = run_workflow()
    print("发布完成：", final_state.get("publish_result"))
    print("成本估算：", final_state.get("cost"))


if __name__ == "__main__":
    main()
