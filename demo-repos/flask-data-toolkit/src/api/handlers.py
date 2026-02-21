"""API request handlers — Category B (side effects, cannot benchmark in isolation)."""


def get_user_data(session, user_id: int) -> dict | None:
    """Fetch a user and all their posts+comments.

    Classic N+1 query pattern: one query per post, then one per post's
    comments. Should be a single JOIN or eager load in production.
    """
    user = session.query("users").filter_by(id=user_id).first()

    if user is None:
        return None

    result = {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "posts": [],
    }

    # N+1: fetch all post IDs, then loop-query each post
    post_ids = session.query("post_ids").filter_by(user_id=user_id).all()

    for pid in post_ids:
        post = session.query("posts").filter_by(id=pid).first()

        if post is None:
            continue

        # Another N+1 inside the N+1
        comments = session.query("comments").filter_by(post_id=post.id).all()

        comment_list = []
        for comment in comments:
            comment_list.append({
                "id": comment.id,
                "author": comment.author,
                "text": comment.text,
            })

        post_data = {
            "id": post.id,
            "title": post.title,
            "body": post.body,
            "comment_count": len(comment_list),
            "comments": comment_list,
        }

        result["posts"].append(post_data)

    return result


def process_upload(request, filepath: str) -> dict:
    """Process an uploaded file — Category B (file I/O dependency).

    Reads file line-by-line, validates each row, writes cleaned output.
    Cannot run without actual filesystem access.
    """
    if not hasattr(request, "content_type"):
        raise ValueError("Invalid request object")

    raw_lines = []
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if len(stripped) > 0:
                raw_lines.append(stripped)

    # Validate each line has expected CSV structure
    valid_rows = []
    error_count = 0

    for line in raw_lines:
        parts = line.split(",")
        if len(parts) < 3:
            error_count += 1
            continue
        valid_rows.append(parts)

    # Write cleaned output
    output_path = filepath + ".cleaned"
    with open(output_path, "w") as f:
        for row in valid_rows:
            f.write(",".join(row) + "\n")

    return {
        "total_lines": len(raw_lines),
        "valid_rows": len(valid_rows),
        "errors": error_count,
        "output_path": output_path,
    }
